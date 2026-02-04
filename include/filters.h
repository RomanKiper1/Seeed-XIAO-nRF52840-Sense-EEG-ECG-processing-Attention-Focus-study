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

#endif  // FILTERS_H
