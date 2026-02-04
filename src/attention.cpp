#include "attention.h"

/**
 * @brief Construct with default score.
 */
AttentionEngine::AttentionEngine() : score_(0.0f) {}

/**
 * @brief Reset internal state (stub).
 */
void AttentionEngine::reset() {
    score_ = 0.0f;
}

/**
 * @brief Update attention score using latest data (stub).
 */
void AttentionEngine::update(const SignalBuffer& buffer, const FeatureVector& features) {
    (void)buffer;
    (void)features;
    // TODO: Compute attention score from features.
}

/**
 * @brief Get current attention score (stub).
 */
float AttentionEngine::getScore() const {
    return score_;
}
