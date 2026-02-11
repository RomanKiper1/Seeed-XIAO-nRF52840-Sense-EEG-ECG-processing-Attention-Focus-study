#include <Arduino.h>

#include "attention.h"
#include "ble_receiver.h"
#include "features.h"
#include "filters.h"
#include "ganglion_parser.h"
#include "memory_manager.h"
#include "signal_buffer.h"

namespace {
constexpr size_t kBufferCapacity = 256;
constexpr size_t kWindowSize = 16;

BleReceiver g_receiver;
GanglionPacketParser g_parser;
SignalBuffer g_raw_buffer(kBufferCapacity);
SignalBuffer g_filtered_buffer(kBufferCapacity);
WinsorizedMedianWindowFilter g_window_filter(8.0f, 7);
NlmsReferenceWindowFilter g_nlms_filter(/*reference_channel_index=*/2,
                                        /*taps=*/16,
                                        /*step_size=*/0.06f,
                                        /*epsilon=*/1e-6f,
                                        /*delay_samples=*/2);
WaveletSym8WindowFilter g_wavelet_filter(/*level=*/3, /*threshold_scale=*/1.0f);
DummyFeatureExtractor g_feature_extractor;
AttentionEngine g_attention;

SampleFrame g_raw_window[kWindowSize];
SampleFrame g_filtered_window[kWindowSize];

/**
 * @brief Callback for raw data frames (stub).
 */
void onRawData(const uint8_t* data, size_t length) {
    GanglionPacketParser::PacketType type = g_parser.parseBlePacket(data, length);
    (void)type;

    SampleFrame frame{};
    while (g_parser.popSampleFrame(frame)) {
        g_raw_buffer.pushSample(frame);
    }
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
    // 1) Input formats: BLE (Ganglion 18/19-bit, raw 24-bit, ASCII/impedance)
    g_receiver.poll();

    // 2) Parsing: GanglionPacketParser -> raw buffer
    if (g_raw_buffer.readWindow(kWindowSize, g_raw_window)) {
        // 3) Processing (pre-filter): placeholder for optional steps
        // 4) Viz v1: handled on host (PC)

        // 5) Filtering: winsorization + masked median
        // NOTE: Offline Python currently compares NLMS vs Wavelet(sym8).
        // On MCU these paths are intentionally stubs until memory and latency
        // budgets are validated.
        //
        // Example wiring (currently disabled):
        // g_nlms_filter.processWindow(g_raw_window, kWindowSize, g_filtered_window);
        // g_wavelet_filter.processWindow(g_raw_window, kWindowSize, g_filtered_window);
        g_window_filter.processWindow(g_raw_window, kWindowSize, g_filtered_window);
        g_filtered_buffer.pushSamples(g_filtered_window, kWindowSize);
    }

    // 6) Viz v2: handled on host (PC)
    // 7) Output/logging: handled on host (PC)
    FeatureVector features = g_feature_extractor.compute(g_filtered_buffer);
    g_attention.update(g_filtered_buffer, features);

    delay(10);
}
