#include "ganglion_parser.h"

/**
 * @brief Construct parser with empty state.
 */
GanglionPacketParser::GanglionPacketParser()
    : queue_head_(0),
      queue_tail_(0),
      queue_size_(0),
      last_type_(PacketType::Unknown),
      last_packet_id_(0) {}

/**
 * @brief Parse a single BLE packet.
 */
GanglionPacketParser::PacketType GanglionPacketParser::parseBlePacket(const uint8_t* data,
                                                                      size_t length) {
    if (data == nullptr || length == 0) {
        last_type_ = PacketType::Unknown;
        return last_type_;
    }

    // Ganglion sends 20-byte packets: [packetId][19 bytes payload]
    if (length < 1) {
        last_type_ = PacketType::Unknown;
        return last_type_;
    }

    last_packet_id_ = data[0];

    if (last_packet_id_ == 0) {
        last_type_ = PacketType::Raw24bit;
        // TODO: Decode 24-bit samples (4 channels) into SampleFrame(s).
        return last_type_;
    }

    if (last_packet_id_ >= 1 && last_packet_id_ <= 100) {
        last_type_ = PacketType::Compressed18bit;
        // TODO: Decode 18-bit compressed samples into SampleFrame(s).
        return last_type_;
    }

    if (last_packet_id_ >= 101 && last_packet_id_ <= 200) {
        last_type_ = PacketType::Compressed19bit;
        // TODO: Decode 19-bit compressed samples into SampleFrame(s).
        return last_type_;
    }

    if (last_packet_id_ >= 201 && last_packet_id_ <= 205) {
        last_type_ = PacketType::ImpedanceAscii;
        // TODO: Parse ASCII impedance values from payload.
        return last_type_;
    }

    if (last_packet_id_ == 206) {
        last_type_ = PacketType::AsciiMessagePart;
        return last_type_;
    }

    if (last_packet_id_ == 207) {
        last_type_ = PacketType::AsciiMessageEnd;
        return last_type_;
    }

    last_type_ = PacketType::Unknown;
    return last_type_;
}

/**
 * @brief Pop one parsed SampleFrame (if available).
 */
bool GanglionPacketParser::popSampleFrame(SampleFrame& out_frame) {
    if (queue_size_ == 0) {
        return false;
    }
    out_frame = queue_[queue_head_];
    queue_head_ = (queue_head_ + 1) % kQueueCapacity;
    queue_size_--;
    return true;
}

/**
 * @brief Last detected packet type.
 */
GanglionPacketParser::PacketType GanglionPacketParser::lastPacketType() const {
    return last_type_;
}

/**
 * @brief Last packet ID byte.
 */
uint8_t GanglionPacketParser::lastPacketId() const {
    return last_packet_id_;
}

void GanglionPacketParser::pushFrame(const SampleFrame& frame) {
    if (queue_size_ >= kQueueCapacity) {
        // Drop newest frame if queue is full (stub behavior).
        return;
    }
    queue_[queue_tail_] = frame;
    queue_tail_ = (queue_tail_ + 1) % kQueueCapacity;
    queue_size_++;
}
