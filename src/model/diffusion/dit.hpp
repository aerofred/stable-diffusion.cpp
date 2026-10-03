#ifndef __SD_MODEL_DIFFUSION_DIT_HPP__
#define __SD_MODEL_DIFFUSION_DIT_HPP__

#include <algorithm>
#include <cstdlib>
#include <string>

#include "core/ggml_extend.h"
#include "core/ggml_runner.h"

namespace DiT {
    inline ggml_tensor* patchify(ggml_context* ctx,
                                 ggml_tensor* x,
                                 int pw,
                                 int ph,
                                 bool patch_last = true) {
        // x: [N, C, H, W]
        // return: [N, h*w, C*ph*pw] if patch_last else [N, h*w, ph*pw*C]
        int64_t N = x->ne[3];
        int64_t C = x->ne[2];
        int64_t H = x->ne[1];
        int64_t W = x->ne[0];
        int64_t h = H / ph;
        int64_t w = W / pw;

        GGML_ASSERT(h * ph == H && w * pw == W);

        x = ggml_reshape_4d(ctx, x, pw, w, ph, h * C * N);     // [N*C*h, ph, w, pw]
        x = ggml_cont(ctx, ggml_permute(ctx, x, 0, 2, 1, 3));  // [N*C*h, w, ph, pw]
        x = ggml_reshape_4d(ctx, x, pw * ph, w * h, C, N);     // [N, C, h*w, ph*pw]
        if (patch_last) {
            x = ggml_cont(ctx, ggml_permute(ctx, x, 0, 2, 1, 3));  // [N, h*w, C, ph*pw]
            x = ggml_reshape_3d(ctx, x, pw * ph * C, w * h, N);    // [N, h*w, C*ph*pw]
        } else {
            x = ggml_cont(ctx, ggml_ext_torch_permute(ctx, x, 2, 0, 1, 3));  // [N, h*w, C, ph*pw]
            x = ggml_reshape_3d(ctx, x, C * pw * ph, w * h, N);              // [N, h*w, ph*pw*C]
        }
        return x;
    }

    inline ggml_tensor* unpatchify(ggml_context* ctx,
                                   ggml_tensor* x,
                                   int64_t h,
                                   int64_t w,
                                   int ph,
                                   int pw,
                                   bool patch_last = true) {
        // x: [N, h*w, C*ph*pw] if patch_last else [N, h*w, ph*pw*C]
        // return: [N, C, H, W]
        int64_t N = x->ne[2];
        int64_t C = x->ne[0] / ph / pw;
        int64_t H = h * ph;
        int64_t W = w * pw;

        GGML_ASSERT(C * ph * pw == x->ne[0]);

        if (patch_last) {
            x = ggml_reshape_4d(ctx, x, pw * ph, C, w * h, N);     // [N, h*w, C, ph*pw]
            x = ggml_cont(ctx, ggml_permute(ctx, x, 0, 2, 1, 3));  // [N, C, h*w, ph*pw]
        } else {
            x = ggml_reshape_4d(ctx, x, C, pw * ph, w * h, N);     // [N, h*w, ph*pw, C]
            x = ggml_cont(ctx, ggml_permute(ctx, x, 2, 0, 1, 3));  // [N, C, h*w, ph*pw]
        }

        x = ggml_reshape_4d(ctx, x, pw, ph, w, h * C * N);     // [N*C*h, w, ph, pw]
        x = ggml_cont(ctx, ggml_permute(ctx, x, 0, 2, 1, 3));  // [N*C*h, ph, w, pw]
        x = ggml_reshape_4d(ctx, x, W, H, C, N);               // [N, C, h*ph, w*pw]

        return x;
    }

    inline ggml_tensor* pad_to_patch_size(GGMLRunnerContext* ctx,
                                          ggml_tensor* x,
                                          int ph,
                                          int pw) {
        int64_t W = x->ne[0];
        int64_t H = x->ne[1];

        int pad_h = (ph - H % ph) % ph;
        int pad_w = (pw - W % pw) % pw;
        x         = ggml_ext_pad(ctx->ggml_ctx, x, pad_w, pad_h, 0, 0, ctx->circular_x_enabled, ctx->circular_y_enabled);
        return x;
    }

    inline ggml_tensor* pad_and_patchify(GGMLRunnerContext* ctx,
                                         ggml_tensor* x,
                                         int ph,
                                         int pw,
                                         bool patch_last = true) {
        x = pad_to_patch_size(ctx, x, ph, pw);
        x = patchify(ctx->ggml_ctx, x, ph, pw, patch_last);
        return x;
    }

    inline ggml_tensor* unpatchify_and_crop(ggml_context* ctx,
                                            ggml_tensor* x,
                                            int64_t H,
                                            int64_t W,
                                            int ph,
                                            int pw,
                                            bool patch_last = true) {
        int pad_h = (ph - H % ph) % ph;
        int pad_w = (pw - W % pw) % pw;
        int64_t h = ((H + pad_h) / ph);
        int64_t w = ((W + pad_w) / pw);
        x         = unpatchify(ctx, x, h, w, ph, pw, patch_last);  // [N, C, H + pad_h, W + pad_w]
        x         = ggml_ext_slice(ctx, x, 1, 0, H);               // [N, C, H, W + pad_w]
        x         = ggml_ext_slice(ctx, x, 0, 0, W);               // [N, C, H, W]
        return x;
    }

    inline ggml_tensor* patchify_3d(ggml_context* ctx,
                                    ggml_tensor* x,
                                    int pt,
                                    int ph,
                                    int pw,
                                    int64_t N       = 1,
                                    bool patch_last = true) {
        // x: [N*C, T, H, W]
        // return: [N, t_len*h_len*w_len, C*pt*ph*pw] if patch_last else [N, t_len*h_len*w_len, C*pt*ph*pw] or [N, t_len*h_len*w_len, pt*ph*pw*C]
        int64_t C     = x->ne[3] / N;
        int64_t T     = x->ne[2];
        int64_t H     = x->ne[1];
        int64_t W     = x->ne[0];
        int64_t t_len = T / pt;
        int64_t h_len = H / ph;
        int64_t w_len = W / pw;

        GGML_ASSERT(C * N == x->ne[3]);
        GGML_ASSERT(t_len * pt == T && h_len * ph == H && w_len * pw == W);

        x = ggml_reshape_4d(ctx, x, pw * w_len, ph * h_len, pt, t_len * C * N);  // [N*C*t_len, pt, h_len*ph, w_len*pw]
        x = ggml_ext_cont(ctx, ggml_ext_torch_permute(ctx, x, 0, 2, 1, 3));      // [N*C*t_len, h_len*ph, pt, w_len*pw]
        x = ggml_reshape_4d(ctx, x, pw * w_len, pt, ph, h_len * t_len * C * N);  // [N*C*t_len*h_len, ph, pt, w_len*pw]
        x = ggml_ext_cont(ctx, ggml_ext_torch_permute(ctx, x, 0, 2, 1, 3));      // [N*C*t_len*h_len, pt, ph, w_len*pw]
        x = ggml_reshape_4d(ctx, x, pw, w_len, ph * pt, h_len * t_len * C * N);  // [N*C*t_len*h_len, pt*ph, w_len, pw]
        x = ggml_ext_cont(ctx, ggml_ext_torch_permute(ctx, x, 0, 2, 1, 3));      // [N*C*t_len*h_len, w_len, pt*ph, pw]
        x = ggml_reshape_4d(ctx, x, pw * ph * pt, w_len * h_len * t_len, C, N);  // [N, C, t_len*h_len*w_len, pt*ph*pw]
        if (patch_last) {
            x = ggml_ext_cont(ctx, ggml_ext_torch_permute(ctx, x, 0, 2, 1, 3));  // [N, t_len*h_len*w_len, C, pt*ph*pw]
        } else {
            x = ggml_ext_cont(ctx, ggml_ext_torch_permute(ctx, x, 2, 0, 1, 3));  // [N, t_len*h_len*w_len, pt*ph*pw, C]
        }
        // [N, t_len*h_len*w_len, C*pt*ph*pw] or [N, t_len*h_len*w_len, pt*ph*pw*C]
        x = ggml_reshape_4d(ctx, x, pw * ph * pt * C, w_len * h_len * t_len, N, 1);
        return x;
    }

    inline ggml_tensor* unpatchify_3d(ggml_context* ctx,
                                      ggml_tensor* x,
                                      int64_t t_len,
                                      int64_t h_len,
                                      int64_t w_len,
                                      int pt,
                                      int ph,
                                      int pw,
                                      bool patch_last = true) {
        // x: [N, t_len*h_len*w_len, C*pt*ph*pw] if patch_last else [N, t_len*h_len*w_len, pt*ph*pw*C]
        // return: [N*C, t_len*pt, h_len*ph, w_len*pw]
        int64_t N = x->ne[2];
        int64_t C = x->ne[0] / pt / ph / pw;

        GGML_ASSERT(C * pt * ph * pw == x->ne[0]);

        if (patch_last) {
            x = ggml_reshape_4d(ctx, x, pw * ph * pt, C, w_len * h_len * t_len, N);  // [N, t_len*h_len*w_len, C, pt*ph*pw]
            x = ggml_ext_cont(ctx, ggml_ext_torch_permute(ctx, x, 0, 2, 1, 3));      // [N, C, t_len*h_len*w_len, pt*ph*pw]
        } else {
            x = ggml_reshape_4d(ctx, x, C, pw * ph * pt, w_len * h_len * t_len, N);  // [N, t_len*h_len*w_len, pt*ph*pw, C]
            x = ggml_ext_cont(ctx, ggml_ext_torch_permute(ctx, x, 1, 2, 0, 3));      // [N, C, t_len*h_len*w_len, pt*ph*pw]
        }

        x = ggml_reshape_4d(ctx, x, pw, ph * pt, w_len, h_len * t_len * C * N);  // [N*C*t_len*h_len, w_len, pt*ph, pw]
        x = ggml_ext_cont(ctx, ggml_ext_torch_permute(ctx, x, 0, 2, 1, 3));      // [N*C*t_len*h_len, pt*ph, w_len, pw]
        x = ggml_reshape_4d(ctx, x, pw * w_len, ph, pt, h_len * t_len * C * N);  // [N*C*t_len*h_len, pt, ph, w_len*pw]
        x = ggml_ext_cont(ctx, ggml_ext_torch_permute(ctx, x, 0, 2, 1, 3));      // [N*C*t_len*h_len, ph, pt, w_len*pw]
        x = ggml_reshape_4d(ctx, x, pw * w_len, pt, ph * h_len, t_len * C * N);  // [N*C*t_len, h_len*ph, pt, w_len*pw]
        x = ggml_ext_cont(ctx, ggml_ext_torch_permute(ctx, x, 0, 2, 1, 3));      // [N*C*t_len, pt, h_len*ph, w_len*pw]
        x = ggml_reshape_4d(ctx, x, pw * w_len, ph * h_len, pt * t_len, C * N);  // [N*C, t_len*pt, h_len*ph, w_len*pw]
        return x;
    }

    // Tokens per range for a stage whose f32 temporaries take bytes_per_token
    // per token: <0 sizes ranges for about 256 MiB of temporaries, 0 disables
    // chunking, >0 is an explicit token count.
    inline int64_t token_chunk_size(int64_t setting, int64_t bytes_per_token, int64_t n_tokens) {
        if (setting == 0) {
            return n_tokens;
        }
        if (setting > 0) {
            return std::min(setting, n_tokens);
        }
        // At most 8 ranges per stage: every range adds graph nodes, and
        // beyond that the saving per extra range is small.
        constexpr int64_t max_ranges = 8;
        int64_t tokens               = (int64_t(256) << 20) / std::max<int64_t>(bytes_per_token, 1);
        tokens                       = std::max<int64_t>(1024, tokens - tokens % 256);
        tokens                       = std::max<int64_t>(tokens, (n_tokens + max_ranges - 1) / max_ranges);
        return std::min(tokens, n_tokens);
    }

    // Writes token ranges of a [dim, tokens] activation into `out` (contiguous,
    // owned by the caller) with chained in-place sets, so the result stays one
    // tensor whose root is the chain's last node (what a graph cut caches).
    // GGML_OP_SET keeps its destination offset in an int32 op param and asserts
    // offset < 1 GiB, so ranges past that bound go into a second chain built on
    // suffix views of `out`, which restart the offset at zero. A suffix view
    // cannot grow back to the full tensor, so `finish` joins the chains with a
    // one-element set on the full chain whose value is routed through the
    // suffix chain: the element is rewritten with its own value and the result
    // now depends on every range. Below 1 GiB the graph is a single chain.
    struct TokenRangeWriter {
        TokenRangeWriter(ggml_context* ctx, ggml_tensor* out)
            : ctx_(ctx), out_(out), head_(out) {
            GGML_ASSERT(ggml_is_contiguous(out) && out->ne[2] == 1 && out->ne[3] == 1);
            GGML_ASSERT(out->nb[1] < max_set_offset);
        }

        // Writes `y` (count * dim elements) over tokens [start, start + count).
        // Ranges past the 1 GiB bound must arrive in increasing token order.
        // Returns the set node, which callers may probe for backend support.
        ggml_tensor* write(ggml_tensor* y, int64_t start, int64_t count) {
            GGML_ASSERT(ggml_nelements(y) == count * out_->ne[0]);
            const size_t nb1 = out_->nb[1];
            const size_t nb2 = nb1 * static_cast<size_t>(count);
            size_t offset    = static_cast<size_t>(start) * nb1;
            if (offset < max_set_offset) {
                head_ = ggml_set_inplace(ctx_, head_, y, nb1, nb2, nb2, offset);
                return head_;
            }
            if (tail_ != nullptr) {
                GGML_ASSERT(start >= tail_start_);
                offset -= static_cast<size_t>(tail_start_) * nb1;
            }
            if (tail_ == nullptr || offset >= max_set_offset) {
                tail_       = ggml_view_2d(ctx_, tail_ != nullptr ? tail_ : out_, out_->ne[0], out_->ne[1] - start, nb1, offset);
                tail_start_ = start;
                offset      = 0;
            }
            tail_ = ggml_set_inplace(ctx_, tail_, y, nb1, nb2, nb2, offset);
            return tail_;
        }

        ggml_tensor* finish() {
            if (tail_ == nullptr) {
                return head_;
            }
            const size_t nb1 = out_->nb[1];
            auto first       = ggml_view_1d(ctx_, head_, 1, 0);
            auto last        = ggml_view_1d(ctx_, tail_, 1, 0);
            auto join        = ggml_view_1d(ctx_, ggml_concat(ctx_, first, last, 0), 1, 0);
            head_            = ggml_set_inplace(ctx_, head_, join, nb1, nb1, nb1, 0);
            tail_            = nullptr;
            return head_;
        }

    private:
        static constexpr size_t max_set_offset = static_cast<size_t>(1) << 30;

        ggml_context* ctx_;
        ggml_tensor* out_;
        ggml_tensor* head_;
        ggml_tensor* tail_  = nullptr;
        int64_t tail_start_ = 0;
    };

    // Parses the `token_chunk` model arg: auto (-1), 0 (disabled) or a token count.
    inline int64_t parse_token_chunk_arg(const std::string& value, const char* model, int64_t fallback) {
        if (value == "auto") {
            return -1;
        }
        char* end        = nullptr;
        long long parsed = std::strtoll(value.c_str(), &end, 10);
        if (!value.empty() && end != nullptr && *end == '\0' && parsed >= 0) {
            return parsed;
        }
        LOG_WARN("ignoring invalid %s model arg 'token_chunk=%s' (expected auto, 0 or a token count)",
                 model, value.c_str());
        return fallback;
    }
}  // namespace DiT

#endif  // __SD_MODEL_DIFFUSION_DIT_HPP__
