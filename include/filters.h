#ifndef FILTERS_H
#define FILTERS_H

#include <cstddef>
#include "signal_buffer.h"

/**
 * @brief Abstract filter interface (stub).
 *
 * SOLID: Open/Closed (extend by new filter types).
 */
class IFilter {
public:
    virtual ~IFilter() = default;

    /**
     * @brief Reset internal filter state (stub).
     */
    virtual void reset() = 0;

    /**
     * @brief Process one sample and return filtered value (stub).
     */
    virtual float process(float sample) = 0;
};

/**
 * @brief Pass-through filter (stub).
 *
 * Responsibility: provide a minimal concrete filter example.
 */
class PassThroughFilter final : public IFilter {
public:
    /**
     * @brief Reset filter state (no state in stub).
     */
    void reset() override;

    /**
     * @brief Return input without changes (stub).
     */
    float process(float sample) override;
};

/**
 * @brief Abstract window filter interface (stub).
 *
 * Responsibility: process a block/window of samples.
 */
class IWindowFilter {
public:
    virtual ~IWindowFilter() = default;

    /**
     * @brief Reset internal filter state (stub).
     */
    virtual void reset() = 0;

    /**
     * @brief Process a window of frames and write to output (stub).
     * @param in_frames Input frames array.
     * @param frame_count Number of frames in the window.
     * @param out_frames Output frames array (same size as input).
     */
    virtual void processWindow(const SampleFrame* in_frames,
                               size_t frame_count,
                               SampleFrame* out_frames) = 0;
};

/**
 * @brief Minimal placeholder window filter (stub).
 */
class DummyWindowFilter final : public IWindowFilter {
public:
    /**
     * @brief Reset filter state (no state in stub).
     */
    void reset() override;

    /**
     * @brief Copy input to output (stub).
     */
    void processWindow(const SampleFrame* in_frames,
                       size_t frame_count,
                       SampleFrame* out_frames) override;
};

/** Channel index for ECG (channels[2] = eeg3). */
constexpr size_t kECGChannelIndex = 2;

/**
 * @brief IIR Bandpass filter (1-45 Hz, 200 Hz fs, Butterworth order 4).
 *
 * Processes all 4 channels. Placed first in pipeline before WinsorizedMedian/NLMS/Wavelet.
 */
class BandpassWindowFilter final : public IWindowFilter {
public:
    BandpassWindowFilter(float low_hz, float high_hz, float sampling_rate);

    void reset() override;
    void processWindow(const SampleFrame* in_frames,
                       size_t frame_count,
                       SampleFrame* out_frames) override;

private:
    float low_hz_;
    float high_hz_;
    float fs_;
    static constexpr size_t kNumBiquads = 4;
    static constexpr size_t kNumChannels = 4;
    float state_[kNumChannels][kNumBiquads][2];  // w[n-1], w[n-2] per biquad per channel
};

/**
 * @brief Window filter with winsorization and masked median smoothing.
 *
 * Steps:
 * 1) Estimate robust scale: sigma = 1.4826 * MAD
 * 2) Winsorize (clip) to +/- c*sigma
 * 3) Apply median filter only where clipping occurred
 */
class WinsorizedMedianWindowFilter final : public IWindowFilter {
public:
    /**
     * @brief Construct with clip factor and median kernel size.
     */
    WinsorizedMedianWindowFilter(float clip_factor, size_t kernel_size);

    /**
     * @brief Reset filter state (no state in stub).
     */
    void reset() override;

    /**
     * @brief Apply winsorization + masked median filter (stub).
     */
    void processWindow(const SampleFrame* in_frames,
                       size_t frame_count,
                       SampleFrame* out_frames) override;

private:
    float clip_factor_;
    size_t kernel_size_;
};

/**
 * @brief Stub for reference-based NLMS ECG artifact cancellation.
 *
 * Design intent (SOLID):
 * - Single responsibility: adaptive reference subtraction stage.
 * - Open for extension: replace internals with MCU-ready fixed-point/stateful NLMS later.
 */
class NlmsReferenceWindowFilter final : public IWindowFilter {
public:
    NlmsReferenceWindowFilter(size_t reference_channel_index,
                              size_t taps,
                              float step_size,
                              float epsilon,
                              size_t delay_samples);

    void reset() override;
    void processWindow(const SampleFrame* in_frames,
                       size_t frame_count,
                       SampleFrame* out_frames) override;

private:
    size_t reference_channel_index_;
    size_t taps_;
    float step_size_;
    float epsilon_;
    size_t delay_samples_;
};

/**
 * @brief Stub for wavelet denoising using sym8 family.
 *
 * Implementation intentionally pass-through for now to keep MCU path lightweight until
 * memory/latency constraints are validated against offline benchmarks.
 */
class WaveletSym8WindowFilter final : public IWindowFilter {
public:
    WaveletSym8WindowFilter(size_t level, float threshold_scale);

    void reset() override;
    void processWindow(const SampleFrame* in_frames,
                       size_t frame_count,
                       SampleFrame* out_frames) override;

private:
    size_t level_;
    float threshold_scale_;
};

#endif  // FILTERS_H
