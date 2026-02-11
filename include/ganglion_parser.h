#ifndef GANGLION_PARSER_H
#define GANGLION_PARSER_H

#include <cstddef>
#include <cstdint>

#include "signal_buffer.h"

/**
 * @brief Ganglion BLE packet parser (stub + packet classification).
 *
 * Responsibility: classify BLE packets and extract SampleFrame data when possible.
 */
class GanglionPacketParser {
public:
    /**
     * @brief Supported packet types from Ganglion BLE stream.
     */
    enum class PacketType {
        Unknown,
        Compressed18bit,
        Compressed19bit,
        Raw24bit,
        ImpedanceAscii,
        AsciiMessagePart,
        AsciiMessageEnd
    };

    /**
     * @brief Construct parser with empty state.
     */
    GanglionPacketParser();

    /**
     * @brief Parse a single BLE packet.
     * @param data Raw BLE packet bytes.
     * @param length Expected 20 bytes for Ganglion packets.
     * @return Detected packet type.
     */
    PacketType parseBlePacket(const uint8_t* data, size_t length);

    /**
     * @brief Pop one parsed SampleFrame (if available).
     * @return True if a frame was returned.
     */
    bool popSampleFrame(SampleFrame& out_frame);

    /**
     * @brief Last detected packet type.
     */
    PacketType lastPacketType() const;

    /**
     * @brief Last packet ID byte.
     */
    uint8_t lastPacketId() const;

private:
    static constexpr size_t kQueueCapacity = 8;

    SampleFrame queue_[kQueueCapacity];
    size_t queue_head_;
    size_t queue_tail_;
    size_t queue_size_;

    PacketType last_type_;
    uint8_t last_packet_id_;

    void pushFrame(const SampleFrame& frame);
};

#endif  // GANGLION_PARSER_H
