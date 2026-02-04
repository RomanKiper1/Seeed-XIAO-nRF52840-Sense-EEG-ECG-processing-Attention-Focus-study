#include "ble_receiver.h"

/**
 * @brief Construct receiver with default values (stub).
 */
BleReceiver::BleReceiver()
    : transport_mode_(TransportMode::Ble), data_callback_(nullptr) {}

/**
 * @brief Initialize receiver resources (stub).
 */
void BleReceiver::begin() {
    // TODO: Initialize BLE stack or USB subsystem.
}

/**
 * @brief Select transport mode (stub).
 */
void BleReceiver::setTransportMode(TransportMode mode) {
    // TODO: Switch between BLE and USB input sources.
    transport_mode_ = mode;
}

/**
 * @brief Register a callback for incoming raw data.
 */
void BleReceiver::setDataCallback(DataCallback callback) {
    data_callback_ = callback;
}

/**
 * @brief Poll receiver state (stub).
 */
void BleReceiver::poll() {
    // TODO: Poll BLE/USB and call data_callback_ when data arrives.
}
