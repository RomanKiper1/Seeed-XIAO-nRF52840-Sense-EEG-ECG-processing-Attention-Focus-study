#include "filters.h"

#include <algorithm>
#include <cmath>

/**
 * @brief Reset filter state (no state in stub).
 */
void PassThroughFilter::reset() {
    // No internal state to reset in stub.
}

/**
 * @brief Return input without changes (stub).
 */
float PassThroughFilter::process(float sample) {
    return sample;
}

/**
 * @brief Reset filter state (no state in stub).
 */
void DummyWindowFilter::reset() {
    // No internal state to reset in stub.
}

/**
 * @brief Copy input to output (stub).
 */
void DummyWindowFilter::processWindow(const SampleFrame* in_frames,
                                      size_t frame_count,
                                      SampleFrame* out_frames) {
    (void)in_frames;
    (void)frame_count;
    (void)out_frames;
    // TODO: Apply window-based filtering and write results to out_frames.
}

namespace {
constexpr size_t kMaxWindow = 64;

float medianInPlace(float* values, size_t count) {
    if (count == 0) {
        return 0.0f;
    }
    for (size_t i = 1; i < count; ++i) {
        float key = values[i];
        size_t j = i;
        while (j > 0 && values[j - 1] > key) {
            values[j] = values[j - 1];
            --j;
        }
        values[j] = key;
    }
    size_t mid = count / 2;
    if ((count % 2) == 1) {
        return values[mid];
    }
    return 0.5f * (values[mid - 1] + values[mid]);
}

float clampFloat(float value, float min_value, float max_value) {
    return std::max(min_value, std::min(max_value, value));
}

/* Precomputed Butterworth bandpass 1-45 Hz, fs=200, order 4 (4 biquads). */
constexpr size_t kBandpassBiquads = 4;
constexpr float kBandpassCoefs[kBandpassBiquads][6] = {
    {0.0629009449f, 0.1258018897f, 0.0629009449f, 1.0f, -0.1971790139f, 0.0514615715f},
    {1.0f, 2.0f, 1.0f, 1.0f, -0.2363252552f, 0.4643304771f},
    {1.0f, -2.0f, 1.0f, 1.0f, -1.9411278602f, 0.9421496071f},
    {1.0f, -2.0f, 1.0f, 1.0f, -1.9758806019f, 0.9768660283f},
};

inline float biquadProcess(float x, float* state, const float* coef) {
    float w = x - coef[4] * state[0] - coef[5] * state[1];
    float y = coef[0] * w + coef[1] * state[0] + coef[2] * state[1];
    state[1] = state[0];
    state[0] = w;
    return y;
}
}  // namespace

BandpassWindowFilter::BandpassWindowFilter(float low_hz, float high_hz, float sampling_rate)
    : low_hz_(low_hz), high_hz_(high_hz), fs_(sampling_rate) {
    reset();
}

void BandpassWindowFilter::reset() {
    for (size_t ch = 0; ch < kNumChannels; ++ch) {
        for (size_t b = 0; b < kNumBiquads; ++b) {
            state_[ch][b][0] = 0.0f;
            state_[ch][b][1] = 0.0f;
        }
    }
}

void BandpassWindowFilter::processWindow(const SampleFrame* in_frames,
                                         size_t frame_count,
                                         SampleFrame* out_frames) {
    if (in_frames == nullptr || out_frames == nullptr || frame_count == 0) {
        return;
    }

    for (size_t ch = 0; ch < BandpassWindowFilter::kNumChannels; ++ch) {
        for (size_t i = 0; i < frame_count; ++i) {
            float x = in_frames[i].channels[ch];
            float y = x;
            for (size_t b = 0; b < BandpassWindowFilter::kNumBiquads; ++b) {
                y = biquadProcess(y, state_[ch][b], kBandpassCoefs[b]);
            }
            out_frames[i].channels[ch] = y;
        }
    }

    for (size_t i = 0; i < frame_count; ++i) {
        out_frames[i].timestamp_ms = in_frames[i].timestamp_ms;
    }
}

WinsorizedMedianWindowFilter::WinsorizedMedianWindowFilter(float clip_factor, size_t kernel_size)
    : clip_factor_(clip_factor), kernel_size_(kernel_size) {}

void WinsorizedMedianWindowFilter::reset() {
    // No internal state to reset in stub.
}

void WinsorizedMedianWindowFilter::processWindow(const SampleFrame* in_frames,
                                                 size_t frame_count,
                                                 SampleFrame* out_frames) {
    if (in_frames == nullptr || out_frames == nullptr || frame_count == 0) {
        return;
    }

    if (frame_count > kMaxWindow) {
        for (size_t i = 0; i < frame_count; ++i) {
            out_frames[i] = in_frames[i];
        }
        return;
    }

    for (size_t i = 0; i < frame_count; ++i) {
        out_frames[i] = in_frames[i];
    }

    size_t kernel = (kernel_size_ < 1) ? 1 : kernel_size_;
    if ((kernel % 2) == 0) {
        kernel += 1;
    }
    size_t half = kernel / 2;

    float channel_values[kMaxWindow];
    float deviations[kMaxWindow];
    float clipped_values[kMaxWindow];
    bool clipped_mask[kMaxWindow];

    for (size_t ch = 0; ch < 4; ++ch) {
        for (size_t i = 0; i < frame_count; ++i) {
            channel_values[i] = in_frames[i].channels[ch];
        }

        float temp_values[kMaxWindow];
        for (size_t i = 0; i < frame_count; ++i) {
            temp_values[i] = channel_values[i];
        }
        float med = medianInPlace(temp_values, frame_count);

        for (size_t i = 0; i < frame_count; ++i) {
            deviations[i] = std::fabs(channel_values[i] - med);
        }

        float temp_dev[kMaxWindow];
        for (size_t i = 0; i < frame_count; ++i) {
            temp_dev[i] = deviations[i];
        }
        float mad = medianInPlace(temp_dev, frame_count);
        float sigma = 1.4826f * mad;

        float limit = clip_factor_ * sigma;
        for (size_t i = 0; i < frame_count; ++i) {
            float v = channel_values[i];
            if (limit > 0.0f && std::fabs(v) > limit) {
                clipped_values[i] = clampFloat(v, -limit, limit);
                clipped_mask[i] = true;
            } else {
                clipped_values[i] = v;
                clipped_mask[i] = false;
            }
        }

        for (size_t i = 0; i < frame_count; ++i) {
            if (!clipped_mask[i]) {
                out_frames[i].channels[ch] = clipped_values[i];
                continue;
            }

            size_t start = (i > half) ? (i - half) : 0;
            size_t end = std::min(frame_count - 1, i + half);
            size_t count = end - start + 1;

            float window_values[kMaxWindow];
            for (size_t j = 0; j < count; ++j) {
                window_values[j] = clipped_values[start + j];
            }
            float local_med = medianInPlace(window_values, count);
            out_frames[i].channels[ch] = local_med;
        }
    }
}

NlmsReferenceWindowFilter::NlmsReferenceWindowFilter(size_t reference_channel_index,
                                                     size_t taps,
                                                     float step_size,
                                                     float epsilon,
                                                     size_t delay_samples)
    : reference_channel_index_(reference_channel_index),
      taps_(taps),
      step_size_(step_size),
      epsilon_(epsilon),
      delay_samples_(delay_samples) {}

void NlmsReferenceWindowFilter::reset() {
    // Stub: no state allocated yet.
}

void NlmsReferenceWindowFilter::processWindow(const SampleFrame* in_frames,
                                              size_t frame_count,
                                              SampleFrame* out_frames) {
    (void)reference_channel_index_;
    (void)taps_;
    (void)step_size_;
    (void)epsilon_;
    (void)delay_samples_;

    if (in_frames == nullptr || out_frames == nullptr || frame_count == 0) {
        return;
    }

    // Stub pass-through until NLMS memory/latency profile is validated on-device.
    for (size_t i = 0; i < frame_count; ++i) {
        out_frames[i] = in_frames[i];
    }
}

WaveletSym8WindowFilter::WaveletSym8WindowFilter(size_t level, float threshold_scale)
    : level_(level), threshold_scale_(threshold_scale) {}

void WaveletSym8WindowFilter::reset() {
    // Stub: no state allocated yet.
}

void WaveletSym8WindowFilter::processWindow(const SampleFrame* in_frames,
                                            size_t frame_count,
                                            SampleFrame* out_frames) {
    (void)level_;
    (void)threshold_scale_;
    if (in_frames == nullptr || out_frames == nullptr || frame_count == 0) {
        return;
    }

    // Stub pass-through until sym8 path is ported with fixed-size buffers.
    for (size_t i = 0; i < frame_count; ++i) {
        out_frames[i] = in_frames[i];
    }
}