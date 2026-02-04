#include "features.h"

/**
 * @brief Reset extractor state (no state in stub).
 */
void DummyFeatureExtractor::reset() {
    // No internal state to reset in stub.
}

/**
 * @brief Return empty feature vector (stub).
 */
FeatureVector DummyFeatureExtractor::compute(const SignalBuffer& buffer) {
    (void)buffer;
    FeatureVector result{};
    result.count = 0;
    return result;
}
