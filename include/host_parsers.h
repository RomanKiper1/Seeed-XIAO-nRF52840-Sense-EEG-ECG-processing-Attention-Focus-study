#ifndef HOST_PARSERS_H
#define HOST_PARSERS_H

#include <cstddef>
#include <cstdint>

/**
 * @brief Host-side parsing stubs for file formats (CSV/EDF/metadata).
 *
 * These parsers are intended for PC/host applications, not MCU firmware.
 */
class CsvParser {
public:
    /**
     * @brief Parse CSV data from a memory buffer (stub).
     */
    bool parseFromBuffer(const uint8_t* data, size_t length);
};

class EdfParser {
public:
    /**
     * @brief Parse EDF data from a memory buffer (stub).
     */
    bool parseFromBuffer(const uint8_t* data, size_t length);
};

class AnnotationParser {
public:
    /**
     * @brief Supported annotation formats (host-side).
     */
    enum class Format {
        Txt,
        Xml,
        Json
    };

    /**
     * @brief Parse annotation data from a memory buffer (stub).
     */
    bool parseFromBuffer(const uint8_t* data, size_t length, Format format);
};

#endif  // HOST_PARSERS_H
