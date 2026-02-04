#ifndef BLE_RECEIVER_H
#define BLE_RECEIVER_H

#include <cstddef>
#include <cstdint>

/**
 * @brief Minimal BLE/USB receiver interface (stub).
 *
 * Responsibility: provide raw frames from BLE or USB sources.
 * SOLID: single responsibility and dependency inversion (callback-based).
 */
class BleReceiver {
public:
    /**
     * @brief Data callback signature for raw frames.
     */
    using DataCallback = void (*)(const uint8_t* data, size_t length);

    /**
     * @brief Available transport modes (stubs, no implementation yet).
     */
    enum class TransportMode {
        Ble,
        Usb
    };

    /**
     * @brief Construct receiver with default transport mode.
     */
    BleReceiver();

    /**
     * @brief Initialize receiver resources (stub).
     */
    void begin();

    /**
     * @brief Select transport mode (stub).
     */
    void setTransportMode(TransportMode mode);

    /**
     * @brief Register a callback for incoming raw data. It accepts a callback function that BleReceiver will call when it receives new raw data (BLE/USB).
     */
    void setDataCallback(DataCallback callback);

    /**
     * @brief Poll receiver state (stub, no I/O yet). Periodically query BLE/USB and save the callback when data is received
     */
    void poll();

private:
    TransportMode transport_mode_;
    DataCallback data_callback_;
};

#endif  // BLE_RECEIVER_H
