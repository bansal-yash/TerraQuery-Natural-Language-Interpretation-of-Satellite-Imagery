// py_qwen3vl.cpp
// Patched qwen_mtmd pybind11 binding
// - preserves legacy infer(image,prompt) (working single-image flow).
// - adds infer_chat(messages) that supports multiple images & text segments.
// - sequential per-block tokenization + correct n_past progression to avoid M-RoPE X<Y errors.
// - when final block is used for generation, append assistant start marker before tokenizing.
// - skips audio-like files (avoids add_media audio errors).
// - safe/free_handle, reset_context.

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <memory>
#include <vector>
#include <string>
#include <sstream>
#include <algorithm>

extern "C" {
#include "../llama.cpp/include/llama.h"
}
#include "../llama.cpp/tools/mtmd/mtmd.h"
#include "../llama.cpp/tools/mtmd/mtmd-helper.h"

namespace py = pybind11;

static void llama_silent_log(ggml_log_level, const char *, void *) {}
static void mtmd_silent_log(ggml_log_level, const char *, void *) {}
static bool g_llama_log_silenced = false;
static bool g_mtmd_log_silenced = false;

struct QwenHandle {
    llama_model *model = nullptr;
    llama_context *ctx = nullptr;
    mtmd_context *vctx = nullptr;
    std::string model_path;
    std::string mmproj_path;
    llama_context_params cp_saved;
    llama_model_params mp_saved;
    bool verbose = false;
};

struct SamplingParams {
    bool do_sample = false;
    int top_k = 40;
    float top_p = 0.9f;
    float temperature = 0.8f;
};

// --------- helpers ----------
static void free_handle_native(QwenHandle *h) {
    if (!h) return;
    if (h->vctx) mtmd_free(h->vctx);
    if (h->ctx) llama_free(h->ctx);
    if (h->model) llama_model_free(h->model);
    delete h;
}

static bool looks_like_audio(const std::string &p) {
    if (p.size() < 4) return false;
    size_t pos = p.find_last_of('.');
    if (pos == std::string::npos) return false;
    std::string ext = p.substr(pos+1);
    for (auto &c : ext) c = (char)tolower(c);
    return (ext=="wav"||ext=="mp3"||ext=="flac"||ext=="m4a"||ext=="aac"||ext=="ogg");
}

static QwenHandle* load_model_and_mmproj(const std::string &model_path,
                                         const std::string &mmproj_path,
                                         int n_gpu_layers = -1,
                                         int n_threads = 8,
                                         bool verbose = false) {
    if (!g_llama_log_silenced) {
        llama_log_set(llama_silent_log, nullptr);
        g_llama_log_silenced = true;
    }
    if (!g_mtmd_log_silenced) {
        mtmd_helper_log_set(mtmd_silent_log, nullptr);
        g_mtmd_log_silenced = true;
    }
    llama_backend_init();

    llama_model_params mp = llama_model_default_params();
    mp.n_gpu_layers = (n_gpu_layers < 0 ? 9999 : n_gpu_layers);
    mp.use_extra_bufts = true;
    mp.split_mode = LLAMA_SPLIT_MODE_ROW;
    mp.no_host = false;

    size_t ndev = llama_max_devices();
    std::vector<float> tensor_split;
    if (ndev > 0) {
        tensor_split.assign(ndev, 1.0f);
        mp.tensor_split = tensor_split.data();
    } else {
        mp.tensor_split = nullptr;
    }

    QwenHandle *h = new QwenHandle();
    h->model_path = model_path;
    h->mmproj_path = mmproj_path;
    h->verbose = verbose;
    h->mp_saved = mp;

    h->model = llama_model_load_from_file(model_path.c_str(), mp);
    if (!h->model) { delete h; throw std::runtime_error("Failed to load model"); }

    // context params
    llama_context_params cp = llama_context_default_params();
    cp.n_threads = n_threads;
    cp.n_threads_batch = n_threads;
    cp.offload_kqv = true;
    cp.kv_unified = true;
    cp.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_AUTO;
#ifdef LLAMA_HAVE_CP_N_GPU_LAYERS
    cp.n_gpu_layers = mp.n_gpu_layers;
#endif

    h->ctx = llama_init_from_model(h->model, cp);
    if (!h->ctx) { llama_model_free(h->model); delete h; throw std::runtime_error("Failed to init llama context"); }
    h->cp_saved = cp;

    // mtmd
    mtmd_context_params vparams = mtmd_context_params_default();
    vparams.use_gpu = true;
    vparams.n_threads = n_threads;
    vparams.media_marker = mtmd_default_marker();

    h->vctx = mtmd_init_from_file(mmproj_path.c_str(), h->model, vparams);
    if (!h->vctx) { llama_free(h->ctx); llama_model_free(h->model); delete h; throw std::runtime_error("Failed to init mtmd context (mmproj)"); }

    return h;
}

static std::string decode_with_sampler(QwenHandle *h, int max_new_tokens, const SamplingParams &params) {
    if (!h) throw std::runtime_error("Handle is null");
    if (max_new_tokens <= 0) return std::string();

    auto sparams = llama_sampler_chain_default_params();
    struct llama_sampler *sampler = llama_sampler_chain_init(sparams);
    if (!sampler) throw std::runtime_error("Failed to initialize sampler chain");

    if (params.do_sample) {
        std::cerr << "[warn] do_sample=true (top_k=" << params.top_k << ", top_p=" << params.top_p
                  << ", temperature=" << params.temperature << ")\n";
        llama_sampler_chain_add(sampler, llama_sampler_init_top_k(std::max(1, params.top_k)));
        float use_top_p = params.top_p;
        if (use_top_p < 0.0f) use_top_p = 0.0f;
        else if (use_top_p > 1.0f) use_top_p = 1.0f;
        llama_sampler_chain_add(sampler, llama_sampler_init_top_p(use_top_p, 1));
        float use_temp = params.temperature < 0.0f ? 0.0f : params.temperature;
        llama_sampler_chain_add(sampler, llama_sampler_init_temp(use_temp));
    }
     else {
        std::cerr << "[warn] do_sample=false -> greedy decoding (top_k/top_p/temperature ignored)\n";
    }
    llama_sampler_chain_add(sampler, llama_sampler_init_greedy());

    const struct llama_vocab *vocab = llama_model_get_vocab(h->model);
    std::string output;
    for (int step = 0; step < max_new_tokens; ++step) {
        llama_token tok = llama_sampler_sample(sampler, h->ctx, -1);
        if (llama_vocab_is_eog(vocab, tok)) break;
        char piece[1024];
        int32_t piece_len = llama_token_to_piece(vocab, tok, piece, (int32_t)sizeof(piece), 0, true);
        if (piece_len > 0) output.append(piece, piece_len);
        struct llama_batch b = llama_batch_get_one(&tok, 1);
        if (llama_decode(h->ctx, b) != 0) { llama_batch_free(b); break; }
    }

    if (sampler) llama_sampler_free(sampler);
    return output;
}

// ---------- legacy single-image infer (keeps the working logic you had) ----------
static std::string legacy_infer(QwenHandle *h, const std::string &image_path, const std::string &prompt, int requested_n_batch, int max_new_tokens, const SamplingParams &sampling_params) {
    if (!h) throw std::runtime_error("Handle is null");

    mtmd_bitmap *bmp = mtmd_helper_bitmap_init_from_file(h->vctx, image_path.c_str());
    if (!bmp) {
        std::vector<unsigned char> grey(32*32*3,128);
        bmp = mtmd_bitmap_init(32,32,grey.data());
        if (!bmp) throw std::runtime_error("Failed to create fallback bitmap");
    }

    std::string marker = mtmd_default_marker();
    std::string prompt_full;
    prompt_full += "<|im_start|>user\n";
    prompt_full += marker + "\n";
    prompt_full += prompt + "\n";
    prompt_full += "<|im_end|>\n";
    prompt_full += "<|im_start|>assistant\n";

    mtmd_input_chunks *chunks = mtmd_input_chunks_init();
    if (!chunks) { mtmd_bitmap_free(bmp); throw std::runtime_error("mtmd_input_chunks_init failed"); }

    mtmd_input_text txt;
    txt.text = prompt_full.c_str();
    txt.add_special = true;
    txt.parse_special = true;

    const mtmd_bitmap *bmps[1] = { bmp };
    int32_t rc = mtmd_tokenize(h->vctx, chunks, &txt, bmps, 1);
    if (rc != 0) { mtmd_input_chunks_free(chunks); mtmd_bitmap_free(bmp); throw std::runtime_error(std::string("mtmd_tokenize failed: ")+std::to_string(rc)); }

    size_t total_tokens = mtmd_helper_get_n_tokens(chunks);
    int n_batch = requested_n_batch;
    if (n_batch <= 0) n_batch = 64;
    if ((size_t)n_batch > total_tokens) n_batch = (int)std::max<size_t>(1, total_tokens);

    llama_pos new_n_past = 0;
    int last_rc = -1;
    while (n_batch >= 1) {
        rc = mtmd_helper_eval_chunks(h->vctx, h->ctx, chunks,
                                     0 /* n_past */,
                                     0 /* seq_id */,
                                     n_batch /* n_batch */,
                                     true /* logits_last */,
                                     &new_n_past);
        last_rc = rc;
        if (rc == 0) break;
        if (n_batch == 1) break;
        int new_batch = std::max(1, n_batch / 2);
        if (h->verbose) std::cerr << "[warn] mtmd_helper_eval_chunks failed: " << rc << ", retrying with n_batch=" << new_batch << "\n";
        n_batch = new_batch;
    }

    if (last_rc != 0) { mtmd_input_chunks_free(chunks); mtmd_bitmap_free(bmp); throw std::runtime_error(std::string("mtmd_helper_eval_chunks failed after retries: ")+std::to_string(last_rc)); }

    std::string output = decode_with_sampler(h, max_new_tokens, sampling_params);

    mtmd_input_chunks_free(chunks);
    mtmd_bitmap_free(bmp);
    return output;
}

// ---------- building HF-style chat blocks + bitmaps ----------
static void build_blocks_from_messages(const py::list &messages,
                                       std::vector<std::string> &out_blocks,
                                       std::vector<mtmd_bitmap*> &out_bmps,
                                       mtmd_context *vctx,
                                       bool verbose=false) {
    out_blocks.clear();
    out_bmps.clear();
    const std::string marker = mtmd_default_marker();

    for (py::handle ph : messages) {
        if (!py::isinstance<py::dict>(ph)) continue;
        py::dict msg = py::cast<py::dict>(ph);
        std::string role = "user";
        if (msg.contains("role")) {
            try { role = py::cast<std::string>(msg["role"]); } catch(...) { role="user"; }
        }

        if (!msg.contains("content")) {
            out_blocks.push_back("<|im_start|>"+role+"\n\n<|im_end|>\n");
            out_bmps.push_back(nullptr);
            continue;
        }

        py::handle content_h = msg["content"];
        py::list content_list;
        if (py::isinstance<py::list>(content_h) || py::isinstance<py::tuple>(content_h)) content_list = py::cast<py::list>(content_h);
        else if (py::isinstance<py::dict>(content_h)) { content_list.append(content_h); }
        else { continue; }

        if (role != "user") {
            std::ostringstream ss;
            ss << "<|im_start|>" << role << "\n";
            for (py::handle segh : content_list) {
                if (!py::isinstance<py::dict>(segh)) continue;
                py::dict seg = py::cast<py::dict>(segh);
                std::string segtype;
                try { if (seg.contains("type")) segtype = py::cast<std::string>(seg["type"]); } catch(...) { segtype=""; }
                if (segtype == "text" || segtype == "txt") {
                    try { ss << py::cast<std::string>(seg["text"]) << "\n"; } catch(...) {}
                } else if (seg.contains("text")) {
                    try { ss << py::cast<std::string>(seg["text"]) << "\n"; } catch(...) {}
                }
            }
            ss << "<|im_end|>\n";
            out_blocks.push_back(ss.str());
            out_bmps.push_back(nullptr);
            continue;
        }

        // user role: gather images and texts
        std::vector<std::string> img_paths;
        std::vector<std::string> texts;
        for (py::handle segh : content_list) {
            if (!py::isinstance<py::dict>(segh)) continue;
            py::dict seg = py::cast<py::dict>(segh);
            std::string segtype;
            try { if (seg.contains("type")) segtype = py::cast<std::string>(seg["type"]); } catch(...) { segtype=""; }
            if (segtype == "image") {
                try { img_paths.push_back(py::cast<std::string>(seg["image"])); } catch(...) { img_paths.push_back(std::string()); }
            } else {
                if (seg.contains("text")) {
                    try { std::string t = py::cast<std::string>(seg["text"]); if (!t.empty()) texts.push_back(t); } catch(...) {}
                }
            }
        }

        if (img_paths.empty()) {
            std::ostringstream ss; ss << "<|im_start|>user\n";
            for (auto &t : texts) ss << t << "\n";
            ss << "<|im_end|>\n";
            out_blocks.push_back(ss.str());
            out_bmps.push_back(nullptr);
            continue;
        }

        // create one block per image; attach texts to last image block
        for (size_t i=0;i<img_paths.size();++i) {
            std::ostringstream ss;
            ss << "<|im_start|>user\n";
            ss << marker << "\n";
            if (i == img_paths.size() - 1) { for (auto &t : texts) ss << t << "\n"; }
            ss << "<|im_end|>\n";
            out_blocks.push_back(ss.str());

            const std::string &p = img_paths[i];
            if (looks_like_audio(p)) {
                if (verbose) std::cerr << "[warn] skipping audio-like media: " << p << "\n";
                out_bmps.push_back(nullptr);
            } else {
                mtmd_bitmap *bmp = nullptr;
                if (!p.empty()) bmp = mtmd_helper_bitmap_init_from_file(vctx, p.c_str());
                if (!bmp) {
                    // fallback: we choose nullptr so tokenize doesn't try to add media that might be unsupported
                    if (verbose) std::cerr << "[warn] bitmap load failed for: " << p << " (using no-bmp)\n";
                    out_bmps.push_back(nullptr);
                } else {
                    out_bmps.push_back(bmp);
                }
            }
        }
    }
    if (!out_blocks.empty() && verbose) std::cerr << "[info] built " << out_blocks.size() << " blocks, images=" << out_bmps.size() << "\n";
}

// ---------- evaluate one prepared block; pass current_n_past and return new_n_past ----------
static std::pair<std::string, llama_pos> eval_block(QwenHandle *h,
                                                    const std::string &block_text,
                                                    mtmd_bitmap *bmp,
                                                    int requested_n_batch,
                                                    int max_new_tokens,
                                                    bool logits_last,
                                                    const SamplingParams &sampling_params,
                                                    llama_pos current_n_past) {
    if (!h) throw std::runtime_error("null handle");

    mtmd_input_chunks *chunks = mtmd_input_chunks_init();
    if (!chunks) throw std::runtime_error("mtmd_input_chunks_init failed");

    // If this block will be followed by generation, append assistant start marker
    // so the tokenizer and model see the assistant role token before sampling.
    std::string text_for_tokenize = block_text;
    if (logits_last) {
        text_for_tokenize += std::string("<|im_start|>assistant\n");
    }

    mtmd_input_text txt;
    // IMPORTANT: txt.text must point to a C-string that remains valid until mtmd_tokenize returns.
    txt.text = text_for_tokenize.c_str();
    txt.add_special = true;
    txt.parse_special = true;

    const mtmd_bitmap *bmp_ptr = bmp;
    int bmp_count = bmp ? 1 : 0;

    int32_t rc = mtmd_tokenize(h->vctx, chunks, &txt, bmp_count ? &bmp_ptr : nullptr, bmp_count);
    if (rc != 0) {
        mtmd_input_chunks_free(chunks);
        std::ostringstream ss; ss << "mtmd_tokenize failed: " << rc;
        throw std::runtime_error(ss.str());
    }

    size_t total_tokens = mtmd_helper_get_n_tokens(chunks);
    int n_batch = requested_n_batch > 0 ? requested_n_batch : 64;
    if ((size_t)n_batch > total_tokens) n_batch = (int)std::max<size_t>(1, total_tokens);

    llama_pos new_n_past = current_n_past;
    int last_rc = -1;
    while (n_batch >= 1) {
        int rc_eval = mtmd_helper_eval_chunks(h->vctx, h->ctx, chunks,
                                              (int)current_n_past,
                                              0 /* seq_id */,
                                              n_batch,
                                              logits_last,
                                              &new_n_past);
        last_rc = rc_eval;
        if (rc_eval == 0) break;
        if (h->verbose) std::cerr << "[warn] mtmd_helper_eval_chunks rc=" << rc_eval << " retrying n_batch=" << std::max(1,n_batch/2) << "\n";
        if (n_batch == 1) break;
        n_batch = std::max(1, n_batch/2);
    }

    if (last_rc != 0) {
        mtmd_input_chunks_free(chunks);
        std::ostringstream ss; ss << "mtmd_helper_eval_chunks failed after retries: " << last_rc;
        throw std::runtime_error(ss.str());
    }

    std::string generated;
    if (logits_last) {
        generated = decode_with_sampler(h, max_new_tokens, sampling_params);
    }

    mtmd_input_chunks_free(chunks);
    return std::make_pair(generated, new_n_past);
}

// ---------- pybind module ----------
PYBIND11_MODULE(qwen_mtmd, m) {
    m.doc() = "Qwen3-VL mtmd binding (patched multi-image chat)";

    py::class_<QwenHandle>(m, "QwenHandle").def(py::init<>());

    m.def("load", [](const std::string &model_path, const std::string &mmproj_path, int n_gpu_layers, int n_threads, bool verbose) {
        QwenHandle *h = load_model_and_mmproj(model_path, mmproj_path, n_gpu_layers, n_threads, verbose);
        return py::capsule(h);
    }, py::arg("model_path"), py::arg("mmproj_path"), py::arg("n_gpu_layers")=-1, py::arg("n_threads")=8, py::arg("verbose")=false);

    // legacy single-image infer (exactly like your working file)
    m.def("infer", [](py::capsule handle, const std::string &image_path, const std::string &prompt, int n_batch, int max_new_tokens,
                      bool do_sample, int top_k, float top_p, float temperature) {
        QwenHandle *h = reinterpret_cast<QwenHandle*>(handle.get_pointer()); if (!h) throw std::runtime_error("Invalid handle");
        SamplingParams sampling_params;
        sampling_params.do_sample = do_sample;
        sampling_params.top_k = top_k;
        sampling_params.top_p = top_p;
        sampling_params.temperature = temperature;
        return legacy_infer(h, image_path, prompt, n_batch, max_new_tokens, sampling_params);
    }, py::arg("handle"), py::arg("image_path"), py::arg("prompt"), py::arg("n_batch")=64, py::arg("max_new_tokens")=128,
       py::arg("do_sample")=false, py::arg("top_k")=40, py::arg("top_p")=0.9f, py::arg("temperature")=0.7f);

    // infer_chat: HF-style messages list
    // m.def("infer_chat", [](py::capsule handle, py::list messages, int n_batch, int max_new_tokens) {
    //     QwenHandle *h = reinterpret_cast<QwenHandle*>(handle.get_pointer()); if (!h) throw std::runtime_error("Invalid handle");
    //     std::vector<std::string> blocks;
    //     std::vector<mtmd_bitmap*> bmps;
    //     build_blocks_from_messages(messages, blocks, bmps, h->vctx, h->verbose);
    //     if (blocks.empty()) return std::string("");
    //     std::string generated_total;
    //     llama_pos current_n_past = 0;
    //     for (size_t i=0;i<blocks.size();++i) {
    //         bool final_block = (i == blocks.size()-1);
    //         mtmd_bitmap *bmp = bmps[i]; // may be nullptr
    //         try {
    //             auto pr = eval_block(h, blocks[i], bmp, n_batch, final_block ? max_new_tokens : 0, final_block, current_n_past);
    //             current_n_past = pr.second;
    //             if (final_block) generated_total = pr.first;
    //         } catch (const std::exception &e) {
    //             // try a single recovery: reset context and retry this block with small batch
    //             if (h->verbose) std::cerr << "[error] block eval failed: " << e.what() << " — attempting reset & retry\n";
    //             if (h->ctx) llama_free(h->ctx);
    //             h->ctx = llama_init_from_model(h->model, h->cp_saved);
    //             try {
    //                 auto pr2 = eval_block(h, blocks[i], bmp, std::max(1, n_batch/4), final_block ? max_new_tokens : 0, final_block, 0);
    //                 current_n_past = pr2.second;
    //                 if (final_block) generated_total = pr2.first;
    //             } catch (const std::exception &e2) {
    //                 for (auto b : bmps) if (b) mtmd_bitmap_free(b);
    //                 throw;
    //             }
    //         }
    //     }
    //     for (auto b : bmps) if (b) mtmd_bitmap_free(b);
    //     return generated_total;
    // }, py::arg("handle"), py::arg("messages"), py::arg("n_batch")=64, py::arg("max_new_tokens")=128);

        // infer_chat: HF-style messages list — independent per-image generation + combined summary
    m.def("infer_chat", [](py::capsule handle, py::list messages, int n_batch, int max_new_tokens,
                            bool do_sample, int top_k, float top_p, float temperature) {
        QwenHandle *h = reinterpret_cast<QwenHandle*>(handle.get_pointer());
        if (!h) throw std::runtime_error("Invalid handle");

        SamplingParams sampling_params;
        sampling_params.do_sample = do_sample;
        sampling_params.top_k = top_k;
        sampling_params.top_p = top_p;
        sampling_params.temperature = temperature;

        // Build HF-style blocks and bitmaps (you already have this)
        std::vector<std::string> blocks;
        std::vector<mtmd_bitmap*> bmps;
        build_blocks_from_messages(messages, blocks, bmps, h->vctx, h->verbose);
        if (blocks.empty()) return std::string("");

        // We'll generate per-image outputs in fresh context each time to avoid KV accumulation.
        std::vector<std::string> per_image_outputs;
        per_image_outputs.reserve(bmps.size());

        // Choose per-image token budget (adjustable)
        int per_image_max = std::max(64, std::min(128, max_new_tokens / 3));
        if (per_image_max > max_new_tokens) per_image_max = max_new_tokens;

        for (size_t i = 0; i < blocks.size(); ++i) {
            bool is_image_block = (bmps[i] != nullptr);

            if (!is_image_block) {
                // For plain text blocks: we do not keep them in global KV — ignore/skip.
                continue;
            }

            // Reinitialize context to clear KV state (same as reset_context)
            if (h->ctx) llama_free(h->ctx);
            h->ctx = llama_init_from_model(h->model, h->cp_saved);
            if (!h->ctx) {
                // fallback: try to re-init and if fail, throw
                throw std::runtime_error("Failed to re-init llama context before per-image generation");
            }

            // Evaluate this one block, generate per-image description
            try {
                auto pr = eval_block(h, blocks[i], bmps[i], n_batch, per_image_max, true, sampling_params, 0 /* start with 0 n_past */);
                per_image_outputs.push_back(pr.first);
            } catch (const std::exception &e) {
                // try with a smaller batch after re-init
                if (h->verbose) std::cerr << "[warn] per-image generation failed, retrying with smaller batch: " << e.what() << "\n";
                if (h->ctx) llama_free(h->ctx);
                h->ctx = llama_init_from_model(h->model, h->cp_saved);
                auto pr2 = eval_block(h, blocks[i], bmps[i], std::max(1, n_batch/4), per_image_max, true, sampling_params, 0);
                per_image_outputs.push_back(pr2.first);
            }
        }

        // Free bitmaps (we no longer need them)
        for (auto b : bmps) if (b) mtmd_bitmap_free(b);

        // Build a *clean* final prompt from the per-image outputs and generate summary in fresh context.
        std::ostringstream final_user;
        final_user << "<|im_start|>user\n";
        final_user << "Here are concise descriptions of the images, in order. Provide one short combined summary.\n";
        for (size_t i = 0; i < per_image_outputs.size(); ++i) {
            final_user << "[Image " << (i+1) << "] " << per_image_outputs[i] << "\n";
        }
        final_user << "<|im_end|>\n";

        // Reset context before final summary
        if (h->ctx) llama_free(h->ctx);
        h->ctx = llama_init_from_model(h->model, h->cp_saved);
        if (!h->ctx) throw std::runtime_error("Failed to re-init llama context before final summary");

        std::string combined_summary;
        try {
            auto pr = eval_block(h, final_user.str(), nullptr, n_batch, max_new_tokens, true, sampling_params, 0);
            combined_summary = pr.first;
        } catch (const std::exception &e) {
            // fallback: smaller batch
            if (h->verbose) std::cerr << "[warn] final summary failed, retrying with smaller batch: " << e.what() << "\n";
            if (h->ctx) llama_free(h->ctx);
            h->ctx = llama_init_from_model(h->model, h->cp_saved);
            auto pr2 = eval_block(h, final_user.str(), nullptr, std::max(1, n_batch/4), max_new_tokens, true, sampling_params, 0);
            combined_summary = pr2.first;
        }

        // Construct return: per-image outputs + combined summary markers
        std::ostringstream out;
        for (size_t i = 0; i < per_image_outputs.size(); ++i) {
            out << "<|image_description_" << (i+1) << "|>\n" << per_image_outputs[i] << "\n\n";
        }
        out << "<|combined_summary|>\n" << combined_summary;

        return out.str();
    }, py::arg("handle"), py::arg("messages"), py::arg("n_batch")=64, py::arg("max_new_tokens")=128,
       py::arg("do_sample")=false, py::arg("top_k")=40, py::arg("top_p")=0.9f, py::arg("temperature")=0.8f);


    m.def("reset_context", [](py::capsule handle) {
        QwenHandle *h = reinterpret_cast<QwenHandle*>(handle.get_pointer()); if (!h) throw std::runtime_error("Invalid handle");
        if (h->ctx) llama_free(h->ctx);
        h->ctx = llama_init_from_model(h->model, h->cp_saved);
        if (!h->ctx) throw std::runtime_error("Failed to re-init llama context");
    }, py::arg("handle"));

    // safe free_handle binding (idempotent)
    m.def("free_handle", [](py::capsule handle) {
        void *p = handle.get_pointer();
        if (!p) return;
        QwenHandle *h = reinterpret_cast<QwenHandle*>(p);
        free_handle_native(h);
        try { handle.set_pointer(nullptr); } catch(...) {}
    }, py::arg("handle"));
}
