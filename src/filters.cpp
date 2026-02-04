#include "filters.h"

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
