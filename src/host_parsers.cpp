#include "host_parsers.h"

bool CsvParser::parseFromBuffer(const uint8_t* data, size_t length) {
    (void)data;
    (void)length;
    // TODO: Implement CSV parsing in a host application.
    return false;
}

bool EdfParser::parseFromBuffer(const uint8_t* data, size_t length) {
    (void)data;
    (void)length;
    // TODO: Implement EDF parsing in a host application.
    return false;
}

bool AnnotationParser::parseFromBuffer(const uint8_t* data, size_t length, Format format) {
    (void)data;
    (void)length;
    (void)format;
    // TODO: Implement TXT/XML/JSON annotation parsing on host.
    return false;
}
