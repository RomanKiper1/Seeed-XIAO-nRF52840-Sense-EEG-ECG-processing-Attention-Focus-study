#include <Arduino.h>

#include "attention.h"
#include "ble_receiver.h"
#include "features.h"
#include "filters.h"
#include "memory_manager.h"
#include "signal_buffer.h"

namespace {
constexpr size_t kBufferCapacity = 256;
constexpr size_t kWindowSize = 16;

BleReceiver g_receiver;
SignalBuffer g_raw_buffer(kBufferCapacity);
SignalBuffer g_filtered_buffer(kBufferCapacity);
DummyWindowFilter g_window_filter;
DummyFeatureExtractor g_feature_extractor;
AttentionEngine g_attention;

SampleFrame g_raw_window[kWindowSize];
SampleFrame g_filtered_window[kWindowSize];

/**
 * @brief Callback for raw data frames (stub).
 */
void onRawData(const uint8_t* data, size_t length) {
    (void)data;
    (void)length;
    // TODO: Parse raw frames into SampleFrame and push into g_raw_buffer.
}
}  // namespace

/**
 * @brief Arduino setup entry point (stub).
 */
void setup() {
    Serial.begin(115200);
    MemoryManager::init();

    g_receiver.setTransportMode(BleReceiver::TransportMode::Ble);
    g_receiver.setDataCallback(&onRawData);
    g_receiver.begin();
}

/**
 * @brief Arduino loop entry point (stub).
 */
void loop() {
    g_receiver.poll();

    if (g_raw_buffer.readWindow(kWindowSize, g_raw_window)) {
        g_window_filter.processWindow(g_raw_window, kWindowSize, g_filtered_window);
        g_filtered_buffer.pushSamples(g_filtered_window, kWindowSize);
    }

    FeatureVector features = g_feature_extractor.compute(g_filtered_buffer);
    g_attention.update(g_filtered_buffer, features);

    delay(10);
}
