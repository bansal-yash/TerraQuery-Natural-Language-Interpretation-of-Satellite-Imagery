// gpu_test.cpp
// Minimal robust test: model + mmproj + image -> generate (tries to place KV on GPU)
// Build with the compilation command shown below.

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <vector>
#include <string>

extern "C" {
#include "../llama.cpp/include/llama.h"
}
#include "../llama.cpp/tools/mtmd/mtmd.h"
#include "../llama.cpp/tools/mtmd/mtmd-helper.h" // provides helper bitmap/load/eval used by CLI

static void die(const char *msg) {
    std::cerr << msg << "\n";
    std::exit(1);
}

int main(int argc, char **argv) {
    if (argc < 4) {
        std::cerr << "Usage: " << argv[0] << " <model.gguf> <mmproj.gguf> <image.png> [prompt]\n";
        return 1;
    }

    const char *model_path  = argv[1];
    const char *mmproj_path = argv[2];
    const char *image_path  = argv[3];
    const char *user_prompt = (argc >= 5) ? argv[4] : "describe the image";

    // 1) init backend exactly like CLI does
    llama_backend_init();

    // 2) model params: keep n_gpu_layers = -1 (auto). IMPORTANT: do not set no_host = true.
    llama_model_params mp = llama_model_default_params();
    mp.n_gpu_layers = -1;            // auto offload (same as CLI)
    mp.use_extra_bufts = true;       // allow repacking / extra buffers (like CLI)
    mp.split_mode = LLAMA_SPLIT_MODE_ROW; // row split for tensor-parallel style if needed
    mp.no_host = false;              // MUST allow host buffers for hybrid models

    // (optional) equal tensor_split for multi-GPU
    size_t ndev = llama_max_devices();
    std::vector<float> tensor_split;
    if (ndev > 0) {
        tensor_split.assign(ndev, 1.0f);
        mp.tensor_split = tensor_split.data();
    } else {
        mp.tensor_split = nullptr;
    }

    // load model (this will attempt to offload layers according to mp)
    struct llama_model *model = llama_model_load_from_file(model_path, mp);
    if (!model) die("Failed to load model");

    // 3) context params: allow KV offload and prefer unified KV on device
    llama_context_params cp = llama_context_default_params();
    cp.n_threads = 8;
    cp.n_threads_batch = 8;
    cp.offload_kqv = false;               // allow KV cache offload to GPU
    cp.kv_unified  = true;               // prefer unified KV buffer (helps on-device placement)
    cp.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_AUTO;

    struct llama_context *ctx = llama_init_from_model(model, cp);
    if (!ctx) die("Failed to create llama context");

    // 4) mtmd: init mmproj/vision context (ask it to use GPU where possible)
    mtmd_context_params vparams = mtmd_context_params_default();
    vparams.use_gpu = true;
    vparams.n_threads = 8;
    vparams.media_marker = mtmd_default_marker();
    // tune if desired:
    // vparams.image_min_tokens = 1024;
    // vparams.image_max_tokens = 4096;

    mtmd_context *vctx = mtmd_init_from_file(mmproj_path, model, vparams);
    if (!vctx) die("Failed to initialize mtmd context (mmproj)");

    std::cout << "[info] model+mmproj loaded\n";

    // 5) load the image via the mtmd helper (CLI uses this helper which handles stbi resizing/etc)
    mtmd_bitmap *bmp = mtmd_helper_bitmap_init_from_file(vctx, image_path);
    if (!bmp) {
        // fallback: create a small grey image so tokenization still runs (debug mode)
        std::cerr << "[warn] mtmd_helper_bitmap_init_from_file failed, using 32x32 grey fallback\n";
        std::vector<unsigned char> grey(32 * 32 * 3, 128);
        bmp = mtmd_bitmap_init(32, 32, grey.data());
    }
    if (!bmp) die("mtmd_bitmap_init failed (no bitmap available)");

    // 6) build prompt and ensure media marker(s) match the number of images
    std::string marker = mtmd_default_marker(); // typically "<__media__>"
    std::string prompt_full;
    prompt_full += "<|im_start|>user\n";
    prompt_full += marker + "\n";
    prompt_full += std::string(user_prompt) + "\n";
    prompt_full += "<|im_end|>\n";
    prompt_full += "<|im_start|>assistant\n";

    // 7) tokenize prompt + image(s)
    mtmd_input_chunks *chunks = mtmd_input_chunks_init();
    if (!chunks) die("mtmd_input_chunks_init failed");

    mtmd_input_text txt;
    txt.text = prompt_full.c_str();
    txt.add_special = true;
    txt.parse_special = true;

    const mtmd_bitmap *bmps[1] = { bmp };

    int32_t rc = mtmd_tokenize(vctx, chunks, &txt, bmps, 1);
    if (rc != 0) {
        std::fprintf(stderr, "[fatal] mtmd_tokenize failed: %d\n", rc);
        mtmd_input_chunks_free(chunks);
        mtmd_bitmap_free(bmp);
        mtmd_free(vctx);
        llama_free(ctx);
        llama_model_free(model);
        llama_backend_free();
        return 1;
    }

    std::cout << "[info] mtmd_tokenize succeeded, chunks = " << mtmd_input_chunks_size(chunks) << "\n";

    // 8) Evaluate chunks using the helper (CLI does this, it handles ordering/decoding/embeddings)
    //    This wraps: text tokens -> llama_decode, image chunk -> mtmd_encode_chunk -> llama_decode(embeddings)
    llama_pos new_n_past = 0;
    size_t total_tokens = mtmd_helper_get_n_tokens(chunks); // helper from mtmd-helper.cpp
    int32_t chosen_batch = 256; // try 256 (you can try 512 on beefy GPUs)
    if (total_tokens == 0) chosen_batch = 1;
    else chosen_batch = (int32_t) std::min<size_t>((size_t)chosen_batch, total_tokens);

    std::cout << "[info] using n_batch = " << chosen_batch << " for mtmd_helper_eval_chunks (total tokens=" << total_tokens << ")\n";

    rc = mtmd_helper_eval_chunks(vctx, ctx, chunks,
                                  0 /* n_past */,
                                  0 /* seq_id */,
                                  chosen_batch /* n_batch */,
                                  true /* logits_last */,
                                  &new_n_past);
    if (rc != 0) {
        std::fprintf(stderr, "[fatal] mtmd_helper_eval_chunks failed: %d\n", rc);
        mtmd_input_chunks_free(chunks);
        mtmd_bitmap_free(bmp);
        mtmd_free(vctx);
        llama_free(ctx);
        llama_model_free(model);
        llama_backend_free();
        return 1;
    }

    std::cout << "[info] chunks evaluated; new_n_past = " << new_n_past << "\n";

    // cleanup input artifacts
    mtmd_input_chunks_free(chunks);
    mtmd_bitmap_free(bmp);

    // 9) sampler chain and sampling loop (same style as CLI)
    auto sparams = llama_sampler_chain_default_params();
    struct llama_sampler *sampler = llama_sampler_chain_init(sparams);
    llama_sampler_chain_add(sampler, llama_sampler_init_top_k(40));
    llama_sampler_chain_add(sampler, llama_sampler_init_top_p(0.9f, 1));
    llama_sampler_chain_add(sampler, llama_sampler_init_temp(0.8f));
    llama_sampler_chain_add(sampler, llama_sampler_init_greedy());

    const struct llama_vocab *vocab = llama_model_get_vocab(model);
    std::string output;

    for (int step = 0; step < 128; ++step) {
        llama_token tok = llama_sampler_sample(sampler, ctx, -1);
        if (llama_vocab_is_eog(vocab, tok)) {
            std::cout << "[EOG]\n";
            break;
        }

        char piece[1024];
        int32_t piece_len = llama_token_to_piece(vocab, tok, piece, (int32_t)sizeof(piece), 0, true);
        if (piece_len > 0) output.append(piece, piece_len);

        struct llama_batch b = llama_batch_get_one(&tok, 1);
        int rc2 = llama_decode(ctx, b);
        if (rc2 != 0) {
            std::fprintf(stderr, "[warn] llama_decode failed during sampling: %d\n", rc2);
            llama_batch_free(b);
            break;
        }
    }

    std::cout << "=== OUTPUT ===\n" << output << "\n==============\n";

    // final cleanup
    if (sampler) llama_sampler_free(sampler);
    if (vctx) mtmd_free(vctx);
    if (ctx)  llama_free(ctx);
    if (model) llama_model_free(model);
    llama_backend_free();

    return 0;
}
