#pragma once
// Frame predictors for the temporal and lossless codecs. Each operates on n3
// floats; `hist` lists frames oldest first. The arithmetic is spelled out in
// a fixed order, and the library is built without FP contraction, so the
// encoder (predicting from its reconstruction) and the decoder agree to the
// bit.

#include <cstddef>
#include <cstring>

namespace mdz::detail {

inline void predict_forward(const float* const* hist, std::size_t n, std::size_t order, std::size_t n3,
                            float* out) noexcept {
    if (n3 == 0) return;
    std::size_t o = order < n ? order : n;
    if (o == 0) {
        std::memset(out, 0, n3 * sizeof(float));
        return;
    }
    if (o == 2) {
        const float *p = hist[n - 2], *l = hist[n - 1];
        for (std::size_t i = 0; i < n3; ++i) out[i] = 2.0f * l[i] - p[i];
        return;
    }
    if (o == 3) {
        const float *f0 = hist[n - 3], *f1 = hist[n - 2], *f2 = hist[n - 1];
        for (std::size_t i = 0; i < n3; ++i) out[i] = 3.0f * f2[i] - 3.0f * f1[i] + f0[i];
        return;
    }
    if (o == 4) {
        const float *f0 = hist[n - 4], *f1 = hist[n - 3], *f2 = hist[n - 2], *f3 = hist[n - 1];
        for (std::size_t i = 0; i < n3; ++i) out[i] = 4.0f * f3[i] - 6.0f * f2[i] + 4.0f * f1[i] - f0[i];
        return;
    }
    // Order 1, and any order above 4, repeats the last frame.
    std::memcpy(out, hist[n - 1], n3 * sizeof(float));
}

// Linear interpolation between two anchors, for B-frames.
inline void predict_interpolate(const float* before, const float* after, std::size_t t_before, std::size_t t_after,
                                std::size_t t_current, std::size_t n3, float* out) noexcept {
    if (n3 == 0) return;
    if (t_after == t_before) {
        std::memcpy(out, before, n3 * sizeof(float));
        return;
    }
    const float alpha = static_cast<float>(t_current - t_before) / static_cast<float>(t_after - t_before);
    const float one_minus_alpha = 1.0f - alpha;
    for (std::size_t i = 0; i < n3; ++i) out[i] = one_minus_alpha * before[i] + alpha * after[i];
}

}  // namespace mdz::detail
