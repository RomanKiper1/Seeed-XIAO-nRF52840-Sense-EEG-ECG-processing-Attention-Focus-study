#ifndef FEATURES_H
#define FEATURES_H

#include <cstddef>
#include "signal_buffer.h"

/**
 * @brief Simple feature vector container (stub).
 */
struct FeatureVector {
    float values[8];
    size_t count;
};

/**
 * @brief Abstract feature extractor interface (stub).
 *
 * Responsibility: compute features from buffered signals.
 */
class IFeatureExtractor {
public:
    virtual ~IFeatureExtractor() = default;

    /**
     * @brief Reset internal extractor state (stub).
     */
    virtual void reset() = 0;

    /**
     * @brief Compute features from the signal buffer (stub).
     */
    virtual FeatureVector compute(const SignalBuffer& buffer) = 0;
};

/**
 * @brief Minimal placeholder extractor (stub).
 */
class DummyFeatureExtractor final : public IFeatureExtractor {
public:
    /**
     * @brief Reset extractor state (no state in stub).
     */
    void reset() override;

    /**
     * @brief Return empty feature vector (stub).
     */
    FeatureVector compute(const SignalBuffer& buffer) override;
};

#endif  // FEATURES_H
