// gpu_test.cpp
// Minimal robust test: model + mmproj + image -> generate
// Build like:
// g++ gpu_test.cpp \
//   -I../llama.cpp/include \
//   -I../llama.cpp/ggml/include \
//   -I../llama.cpp/tools/stb \
//   -I../llama.cpp/tools/mtmd \
//   -I../llama.cpp \
//   -L../llama.cpp/build/bin \
//   -lllama -lmtmd -lggml-cuda -lggml -ldl -lpthread -O3 -o gpu_test

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
#include "../llama.cpp/tools/mtmd/mtmd-helper.h"

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

    // init ggml / llama backends
    llama_backend_init();

    // --- Model params: similar to CLI for GPU offload and row split ---
    llama_model_params mp = llama_model_default_params();
    mp.n_gpu_layers = -1;       // try full offload (auto) — tune if needed
    mp.use_mmap     = false;    // disable mmap host-only so offload is allowed
    mp.no_host      = true;
    mp.use_extra_bufts = true;
    mp.split_mode   = LLAMA_SPLIT_MODE_ROW;

    // (optional) provide equal tensor_split for all devices
    size_t ndev = llama_max_devices();
    std::vector<float> tensor_split;
    if (ndev > 0) {
        tensor_split.assign(ndev, 1.0f);
        mp.tensor_split = tensor_split.data();
    } else {
        mp.tensor_split = nullptr;
    }

    // Load the model
    struct llama_model *model = llama_model_load_from_file(model_path, mp);
    if (!model) die("Failed to load model");

    // Create a context with GPU KV offload and decent threads
    llama_context_params cp = llama_context_default_params();
    cp.n_threads = 8;
    cp.n_threads_batch = 8;
    cp.offload_kqv = true;              // try to put KV cache on GPU
    cp.kv_unified  = false;
    cp.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_AUTO;

    struct llama_context *ctx = llama_init_from_model(model, cp);
    if (!ctx) die("Failed to create llama context");

    // --- mtmd: initialize mmproj/vision context ---
    mtmd_context_params vparams = mtmd_context_params_default();
    vparams.use_gpu = true;                // ask mmproj to offload to GPU if possible
    vparams.n_threads = 8;
    vparams.media_marker = mtmd_default_marker(); // keep default
    // optionally tune image_min_tokens / image_max_tokens if you know the model's hparams
    // vparams.image_min_tokens = 1024;
    // vparams.image_max_tokens = 4096;

    mtmd_context *vctx = mtmd_init_from_file(mmproj_path, model, vparams);
    if (!vctx) die("Failed to initialize mtmd context (mmproj)");

    // Load image into mtmd bitmap using the helper (CLI uses this helper)
    mtmd::bitmap bmp(mtmd_helper_bitmap_init_from_file(vctx, image_path));
    if (!bmp.ptr) die("Failed to load image via mtmd_helper_bitmap_init_from_file");

    // Build prompt and ensure marker(s) exist for each image
    std::string marker = mtmd_default_marker(); // e.g. "<__media__>"
    std::string prompt_full;

    // The mtmd_tokenize API expects the prompt to contain the marker(s).
    // The CLI wraps the prompt in chat-like special tokens; we keep a simple single-turn prompt.
    // Add the marker exactly once for the single image.
    prompt_full += "<|im_start|>user\n";
    prompt_full += marker + "\n";
    prompt_full += std::string(user_prompt) + "\n";
    prompt_full += "<|im_end|>\n";
    prompt_full += "<|im_start|>assistant\n";

    // Tokenize using mtmd_tokenize (must pass same number of bitmaps as markers)
    mtmd_input_chunks *chunks = mtmd_input_chunks_init();
    if (!chunks) die("mtmd_input_chunks_init failed");

    mtmd_input_text txt;
    txt.text = prompt_full.c_str();
    txt.add_special = true;   // add special tokens like <|im_start|> if model expects them
    txt.parse_special = true; // enable special parsing (markers etc)

    // Note: mtmd::bitmaps::c_ptr() in CLI returns array of pointers; we only have one image
    const mtmd_bitmap *bmps[1] = { bmp.ptr.get() };

    int32_t ret = mtmd_tokenize(vctx, chunks, &txt, bmps, 1);
    if (ret != 0) {
        std::fprintf(stderr, "mtmd_tokenize failed: %d\n", ret);
        mtmd_input_chunks_free(chunks);
        mtmd_bitmap_free(bmp.ptr.release()); // safe free if still exists
        mtmd_free(vctx);
        llama_free(ctx);
        llama_model_free(model);
        llama_backend_free();
        return 1;
    }

    std::cout << "[info] mtmd_tokenize succeeded, chunks = " << mtmd_input_chunks_size(chunks) << "\n";

    // Use helper to evaluate all chunks (this will call llama_decode appropriately for text and call
    // mtmd_encode_chunk + mtmd_helper_decode_image_chunk for image/audio with correct positions etc.)
    llama_pos new_n_past = 0;
    ret = mtmd_helper_eval_chunks(vctx, ctx, chunks, 0 /* n_past */, 0 /* seq_id */, 1 /* n_batch */, true /* logits_last */, &new_n_past);
    if (ret != 0) {
        std::fprintf(stderr, "mtmd_helper_eval_chunks failed: %d\n", ret);
        mtmd_input_chunks_free(chunks);
        mtmd_bitmap_free(bmp.ptr.release());
        mtmd_free(vctx);
        llama_free(ctx);
        llama_model_free(model);
        llama_backend_free();
        return 1;
    }

    std::cout << "[info] chunks evaluated; new_n_past = " << new_n_past << "\n";

    // free the intermediate structures
    mtmd_input_chunks_free(chunks);
    // keep mtmd bitmap alive only if you plan to reuse (we free it)
    mtmd_bitmap_free(bmp.ptr.release());

    // ---- setup sampler chain (simple: top-k/top-p/temp/greedy) ----
    auto sparams = llama_sampler_chain_default_params();
    struct llama_sampler *sampler = llama_sampler_chain_init(sparams);
    llama_sampler_chain_add(sampler, llama_sampler_init_top_k(40));
    llama_sampler_chain_add(sampler, llama_sampler_init_top_p(0.9f, 1));
    llama_sampler_chain_add(sampler, llama_sampler_init_temp(0.8f));
    llama_sampler_chain_add(sampler, llama_sampler_init_greedy());

    const struct llama_vocab *vocab = llama_model_get_vocab(model);

    // sampling loop: produce up to 128 tokens
    std::string output;
    for (int step = 0; step < 128; ++step) {
        llama_token tok = llama_sampler_sample(sampler, ctx, -1);
        if (llama_vocab_is_eog(vocab, tok)) {
            std::cout << "[EOG]\n";
            break;
        }

        char piece[512];
        int32_t piece_len = llama_token_to_piece(vocab, tok, piece, sizeof(piece), 0, true);
        if (piece_len > 0) output.append(piece, piece_len);

        // feed token back to model
        struct llama_batch b = llama_batch_get_one(&tok, 1);
        int rc = llama_decode(ctx, b);
        if (rc != 0) {
            std::fprintf(stderr, "llama_decode failed during sampling: %d\n", rc);
            llama_batch_free(b);
            break;
        }
    }

    std::cout << "=== OUTPUT ===\n" << output << "\n==============\n";

    // cleanup
    if (sampler) llama_sampler_free(sampler);
    if (vctx) mtmd_free(vctx);
    if (ctx)  llama_free(ctx);
    if (model) llama_model_free(model);
    llama_backend_free();

    return 0;
}
