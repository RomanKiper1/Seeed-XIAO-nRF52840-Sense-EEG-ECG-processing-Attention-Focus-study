#include "signal_buffer.h"

/**
 * @brief Construct buffer with desired capacity (stub).
 */
SignalBuffer::SignalBuffer(size_t capacity) : capacity_(capacity) {}

/**
 * @brief Clear all buffered samples (stub).
 */
void SignalBuffer::clear() {
    // TODO: Clear internal ring buffer.
}

/**
 * @brief Push a new sample into the buffer (stub).
 */
void SignalBuffer::pushSample(const SampleFrame& sample) {
    (void)sample;
    // TODO: Store sample in ring buffer.
}

/**
 * @brief Pop the oldest sample (stub).
 */
bool SignalBuffer::popSample(SampleFrame& out_sample) {
    (void)out_sample;
    // TODO: Retrieve sample from ring buffer.
    return false;
}

/**
 * @brief Read a window of frames without removing them (stub).
 */
bool SignalBuffer::readWindow(size_t window_size, SampleFrame* out_frames) const {
    (void)window_size;
    (void)out_frames;
    // TODO: Copy window_size frames into out_frames.
    return false;
}

/**
 * @brief Push a batch of samples (stub).
 */
void SignalBuffer::pushSamples(const SampleFrame* samples, size_t count) {
    (void)samples;
    (void)count;
    // TODO: Push multiple samples into ring buffer.
}

/**
 * @brief Current number of stored samples (stub).
 */
size_t SignalBuffer::size() const {
    // TODO: Return current size of ring buffer.
    return 0;
}

/**
 * @brief Max capacity of the buffer (stub).
 */
size_t SignalBuffer::capacity() const {
    return capacity_;
}
