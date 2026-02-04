#ifndef SIGNAL_BUFFER_H
#define SIGNAL_BUFFER_H

#include <cstddef>
#include <cstdint>

/**
 * @brief Signal sample container (EEG/ECG channels).
 *
 * Channels are placeholders: EEG in 1-2, ECG in 3, channel 4 reserved/noise.
 */
struct SampleFrame {
    uint32_t timestamp_ms;
    float channels[4];
};

/**
 * @brief Simple signal buffer interface (stub).
 *
 * Responsibility: store and provide access to signal frames.
 */
class SignalBuffer {
public:
    /**
     * @brief Construct buffer with desired capacity (stub).
     */
    explicit SignalBuffer(size_t capacity);

    /**
     * @brief Clear all buffered samples (stub).
     */
    void clear();

    /**
     * @brief Push a new sample into the buffer (stub).
     */
    void pushSample(const SampleFrame& sample);

    /**
     * @brief Pop the oldest sample (stub).
     * @return True if a sample was available.
     */
    bool popSample(SampleFrame& out_sample);

    /**
     * @brief Read a window of frames without removing them (stub).
     * @param window_size Number of frames to read.
     * @param out_frames Output array with at least window_size elements.
     * @return True if enough frames were available.
     */
    bool readWindow(size_t window_size, SampleFrame* out_frames) const;

    /**
     * @brief Push a batch of samples (stub).
     */
    void pushSamples(const SampleFrame* samples, size_t count);

    /**
     * @brief Current number of stored samples (stub).
     */
    size_t size() const;

    /**
     * @brief Max capacity of the buffer (stub).
     */
    size_t capacity() const;

private:
    size_t capacity_;
};

#endif  // SIGNAL_BUFFER_H
