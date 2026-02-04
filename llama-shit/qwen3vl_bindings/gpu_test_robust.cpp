// gpu_test_robust.cpp
// Robust minimal test for Qwen3-VL (mmproj + model + GPU offload + sampling).
//
// Notes:
// - Requires llama.cpp built (libs in ../llama.cpp/build/bin)
// - Requires ../llama.cpp/tools/mtmd and mmproj file, and stb_image.h in repo
// - This program ensures the mmproj media marker is inserted into the prompt
//   so mtmd_tokenize sees the same number of bitmaps and markers.

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <vector>
#include <string>

extern "C" {
    #include "../llama.cpp/include/llama.h"
    // stb_image is a C header; include under extern "C"
    #include "stb_image.h"
}
#include "../llama.cpp/tools/mtmd/mtmd.h"

static void die(const char *msg) {
    std::cerr << msg << "\n";
    std::exit(1);
}

static std::vector<unsigned char> load_rgb_from_file(const char *path, int &W, int &H) {
    int channels = 0;
    unsigned char *data = stbi_load(path, &W, &H, &channels, 3);
    if (!data) {
        std::cerr << "stb_image failed to load: " << path << " (stbi_failure_reason: " << stbi_failure_reason() << ")\n";
        return {};
    }
    size_t n = (size_t)W * (size_t)H * 3;
    std::vector<unsigned char> out;
    out.resize(n);
    memcpy(out.data(), data, n);
    stbi_image_free(data);
    return out;
}

int main(int argc, char **argv) {
    if (argc < 4) {
        std::cerr << "Usage: " << argv[0] << " <model.gguf> <mmproj.gguf> <image.png> [prompt]\n";
        return 1;
    }
    const char *model_path = argv[1];
    const char *mmproj_path = argv[2];
    const char *image_path = argv[3];
    std::string base_prompt = (argc >= 5) ? argv[4] : "Describe the image.";

    std::cout << "[info] model: " << model_path << "\n";
    std::cout << "[info] mmproj: " << mmproj_path << "\n";
    std::cout << "[info] image: " << image_path << "\n";

    // 1) Init backend
    llama_backend_init();

    // 2) model params (force GPU offload behavior similar to CLI)
    struct llama_model_params mp = llama_model_default_params();

    mp.n_gpu_layers = -1;      // ask for maximum offload (auto)
    mp.use_mmap = false;       // avoid host-mmap-only mapping
    mp.no_host = true;         // attempt device-only buffers where possible
    mp.use_extra_bufts = true; // allow CUDA buffer optimizations
    mp.split_mode = LLAMA_SPLIT_MODE_ROW;

    // optional: assign equal tensor_split across available devices
    size_t ndev = llama_max_devices();
    std::vector<float> tensor_split;
    if (ndev > 0) {
        tensor_split.assign(ndev, 1.0f);
        mp.tensor_split = tensor_split.data();
    } else {
        mp.tensor_split = nullptr;
    }

    std::cout << "[info] loading model (this will take a while)...\n";
    struct llama_model *model = llama_model_load_from_file(model_path, mp);
    if (!model) die("[error] failed to load model");

    // 3) context params - enable GPU KV offload and flash attn auto
    struct llama_context_params cp = llama_context_default_params();
    cp.n_threads = 8;
    cp.n_threads_batch = 8;
    cp.offload_kqv = true;
    cp.kv_unified = false;
    cp.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_AUTO;

    struct llama_context *ctx = llama_init_from_model(model, cp);
    if (!ctx) die("[error] failed to create context");

    // 4) mtmd params
    struct mtmd_context_params vparams = mtmd_context_params_default();
    vparams.use_gpu = true;
    vparams.n_threads = 8;
    vparams.media_marker = mtmd_default_marker(); // usually "<__media__>" or mmproj-specific
    // Qwen-VL guidance
    vparams.image_min_tokens = 1024;
    vparams.image_max_tokens = 4096;

    std::cout << "[info] initializing mmproj (mtmd)...\n";
    mtmd_context *vctx = mtmd_init_from_file(mmproj_path, model, vparams);
    if (!vctx) die("[error] mtmd_init_from_file failed");

    std::cout << "[info] loading image into RGB buffer via stb_image...\n";
    int W=0, H=0;
    auto rgb = load_rgb_from_file(image_path, W, H);
    if (rgb.empty()) {
        std::cerr << "[warn] couldn't load image; falling back to 32x32 black placeholder\n";
        const int Wb = 32, Hb = 32;
        W = Wb; H = Hb;
        rgb.assign(W*H*3, 0);
    } else {
        std::cout << "[info] image loaded: " << W << "x" << H << "\n";
    }

    mtmd_bitmap *bmp = mtmd_bitmap_init((uint32_t)W, (uint32_t)H, rgb.data());
    if (!bmp) die("[error] mtmd_bitmap_init failed");

    // Build prompt and ensure it includes the media marker.
    // We wrap in a chat-style template like the CLI to match model expectations.
    const char *marker = mtmd_default_marker(); // same as vparams.media_marker
    std::string prompt;
    // if base_prompt already contains marker, preserve it; otherwise insert
    if (base_prompt.find(marker) != std::string::npos) {
        prompt = base_prompt;
    } else {
        // Chat style wrapper + marker
        prompt = std::string("<|im_start|>user\n") + marker + "\n" + base_prompt + "\n<|im_end|>";
    }
    std::cout << "[info] final prompt:\n" << prompt << "\n";

    // Tokenize
    mtmd_input_chunks *chunks = mtmd_input_chunks_init();
    if (!chunks) die("[error] mtmd_input_chunks_init failed");

    mtmd_input_text txt;
    txt.text = prompt.c_str();
    txt.add_special = true;
    txt.parse_special = true;

    const mtmd_bitmap *bmps[1] = { bmp };
    int32_t tok_err = mtmd_tokenize(vctx, chunks, &txt, bmps, 1);
    if (tok_err != 0) {
        // helpful diagnostics
        std::cerr << "[error] mtmd_tokenize failed (code=" << tok_err << ")\n";
        std::cerr << "[hint] number of bitmaps: 1, ensure prompt contains the media marker: " << marker << "\n";
        mtmd_input_chunks_free(chunks);
        mtmd_bitmap_free(bmp);
        mtmd_free(vctx);
        llama_free(ctx);
        llama_model_free(model);
        llama_backend_free();
        return 2;
    }
    std::cout << "[info] mtmd_tokenize succeeded, chunks = " << mtmd_input_chunks_size(chunks) << "\n";

    // Setup sampler chain
    llama_sampler_chain_params sp = llama_sampler_chain_default_params();
    struct llama_sampler *chain = llama_sampler_chain_init(sp);
    llama_sampler_chain_add(chain, llama_sampler_init_top_k(40));
    llama_sampler_chain_add(chain, llama_sampler_init_top_p(0.9f, 1));
    llama_sampler_chain_add(chain, llama_sampler_init_temp(0.8f));
    llama_sampler_chain_add(chain, llama_sampler_init_greedy());

    // Walk chunks
    size_t nch = mtmd_input_chunks_size(chunks);
    for (size_t i = 0; i < nch; ++i) {
        const mtmd_input_chunk *ch = mtmd_input_chunks_get(chunks, i);
        int t = mtmd_input_chunk_get_type(ch);
        if (t == MTMD_INPUT_CHUNK_TYPE_TEXT) {
            size_t ntokens = 0;
            const llama_token *tokens = mtmd_input_chunk_get_tokens_text(ch, &ntokens);
            if (ntokens > 0) {
                struct llama_batch batch = llama_batch_get_one((llama_token*)tokens, (int32_t)ntokens);
                int32_t rc = llama_decode(ctx, batch);
                if (rc < 0) die("[error] llama_decode failed on text chunk");
            }
                } else if (t == MTMD_INPUT_CHUNK_TYPE_IMAGE) {
            // 1) encode the image chunk (runs vision encoder + projector)
            int32_t rc = mtmd_encode_chunk(vctx, ch);
            if (rc != 0) die("[error] mtmd_encode_chunk failed");

            // 2) get pointer to output embeddings (flattened float array)
            float *emb = mtmd_get_output_embd(vctx);
            if (!emb) die("[error] mtmd_get_output_embd returned null");

            // 3) number of tokens vs number of positions:
            size_t n_tokens = mtmd_input_chunk_get_n_tokens(ch);
            size_t n_pos    = mtmd_input_chunk_get_n_pos(ch);

            if (n_tokens == 0) {
                // Many mmproj variants set n_tokens==0 and use n_pos for image positions.
                n_tokens = n_pos;
            }

            if (n_tokens == 0) {
                std::cerr << "[error] mtmd produced zero tokens/positions for image chunk\n";
                die("[error] cannot feed zero-length image embeddings into llama");
            }

            // 4) sanity logging (optional - helpful while debugging)
            int32_t model_embd = llama_model_n_embd(model);
            int32_t proj_dim = mtmd_get_output_embd ? mtmd_get_output_embd(vctx) : model_embd;
            std::cerr << "[debug] image: n_tokens=" << n_tokens << " n_pos=" << n_pos
                      << " model_embd=" << model_embd << " proj_dim=" << proj_dim << "\n";

            // 5) prepare llama batch and copy embeddings
            // llama expects embeddings shaped [n_tokens x model_embd]
            struct llama_batch batch = llama_batch_init((int32_t)n_tokens, model_embd, 1);

            // copy n_tokens * model_embd floats from emb -> batch.embd
            size_t n_floats = (size_t)n_tokens * (size_t)model_embd;
            memcpy(batch.embd, emb, n_floats * sizeof(float));

            // 6) decode
            rc = llama_decode(ctx, batch);
            llama_batch_free(batch);
            if (rc < 0) die("[error] llama_decode on image embeddings failed");
        }

    }

    // cleanup chunks/bitmap
    mtmd_input_chunks_free(chunks);
    mtmd_bitmap_free(bmp);

    std::cout << "[info] context ready. sampling up to 128 tokens...\n";

    const struct llama_vocab *vocab = llama_model_get_vocab(model);
    std::string out;
    for (int step = 0; step < 128; ++step) {
        llama_token tok = llama_sampler_sample(chain, ctx, -1);
        if (llama_vocab_is_eog(vocab, tok)) { std::cout << "[EOG]\n"; break; }
        char buf[1024];
        int32_t n = llama_token_to_piece(vocab, tok, buf, (int32_t)sizeof(buf), 0, true);
        if (n > 0) out.append(buf, n);

        struct llama_batch b = llama_batch_get_one(&tok, 1);
        int rc = llama_decode(ctx, b);
        if (rc < 0) { std::cerr << "[error] llama_decode failed in sampling loop\n"; break; }
    }

    std::cout << "=== OUTPUT ===\n" << out << "\n==============\n";

    // teardown
    if (chain) llama_sampler_free(chain);
    if (vctx) mtmd_free(vctx);
    if (ctx) llama_free(ctx);
    if (model) llama_model_free(model);
    llama_backend_free();

    return 0;
}
