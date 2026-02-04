#ifndef ATTENTION_H
#define ATTENTION_H

#include "features.h"
#include "signal_buffer.h"

/**
 * @brief High-level attention estimation engine (stub).
 *
 * Responsibility: combine features into a single attention score.
 */
class AttentionEngine {
public:
    /**
     * @brief Construct with default score.
     */
    AttentionEngine();

    /**
     * @brief Reset internal state (stub).
     */
    void reset();

    /**
     * @brief Update attention score using latest data (stub).
     */
    void update(const SignalBuffer& buffer, const FeatureVector& features);

    /**
     * @brief Get current attention score (stub).
     */
    float getScore() const;

private:
    float score_;
};

#endif  // ATTENTION_H
