// blank_motor_stop.ino
// Real-time EEG/ECG DSP pipeline for Seeed XIAO nRF52840
// Receives 4-channel data from OpenBCI Ganglion via BLE,
// filters (bandpass + notch), computes attention index and heart rate,
// sends structured binary packets to PC via Serial USB.

#include <ArduinoBLE.h>
#include "arduinoFFT.h"
#include <math.h>
#include <string.h>

// ========================== CONFIGURATION ==========================

const int    MOTOR_PIN          = 1;
const int    NUM_CHANNELS       = 4;
const int    FFT_SIZE           = 256;
const float  SAMPLING_FREQ      = 200.0f;
const float  ATTENTION_THRESHOLD = 0.5f;
const float  GANGLION_UV        = 0.001869917138805f;
const float  DF                 = SAMPLING_FREQ / FFT_SIZE;  // 0.78125 Hz

// EMA smoothing coefficient for attention
const float  EMA_ALPHA          = 0.2f;

// ECG refractory period in samples (200 Hz / 3 ~ 67 samples = 0.33 s)
const int    ECG_REFRACTORY     = 67;

// ========================== BUFFERS ================================

// PRE-FILTER BUFFER (Raw Ring Buffer)
// Holds decoded, uV-scaled samples BEFORE any filtering.
// Written by decompressPacket19bit() after scaling.
// Read by DC removal and the filter pipeline.
float rawBuffer[NUM_CHANNELS][FFT_SIZE];

// POST-FILTER BUFFER (Filtered Ring Buffer)
// Holds samples AFTER bandpass + notch filtering.
// Written by the filter pipeline (bandpass -> notch).
// Read by FFT (ch0, ch1 for attention) and ECG peak detection (ch2).
float filteredBuffer[NUM_CHANNELS][FFT_SIZE];

int bufferHead  = 0;
int sampleCount = 0;

// Accumulated raw values for 19-bit delta decompression
int32_t last_values[NUM_CHANNELS] = {0, 0, 0, 0};

// ========================== FFT ====================================

double vReal[FFT_SIZE];
double vImag[FFT_SIZE];
ArduinoFFT<double> FFT = ArduinoFFT<double>(vReal, vImag, FFT_SIZE, SAMPLING_FREQ);

// ========================== FILTER COEFFICIENTS ====================

// Butterworth bandpass 1-45 Hz, fs=200, order 4 (4 cascaded biquads)
// Ported from Project/src/filters.cpp
const float BP_COEFS[4][6] = {
    {0.0629009449f, 0.1258018897f, 0.0629009449f, 1.0f, -0.1971790139f, 0.0514615715f},
    {1.0f,          2.0f,          1.0f,          1.0f, -0.2363252552f, 0.4643304771f},
    {1.0f,         -2.0f,          1.0f,          1.0f, -1.9411278602f, 0.9421496071f},
    {1.0f,         -2.0f,          1.0f,          1.0f, -1.9758806019f, 0.9768660283f},
};
float bpState[NUM_CHANNELS][4][2];

// Notch 50 Hz, Q=30, fs=200  (scipy.signal.iirnotch)
// b = [0.9744823, 0.0, 0.9744823],  a = [1.0, 0.0, 0.9489646]
const float NOTCH_COEFS[6] = {
    0.9744822834f, 0.0f, 0.9744822834f,
    1.0f,          0.0f, 0.9489645667f
};
float notchState[NUM_CHANNELS][2];

// ========================== OUTPUT STATE ===========================

float smoothedAttention = 0.5f;
uint8_t seqNum = 0;

// ========================== BLE PERIPHERAL ===========================

BLEService bioService("19B10000-E8F2-537E-4F6C-D104768A1214");
BLECharacteristic bioChar("19B10001-E8F2-537E-4F6C-D104768A1214",
                          BLERead | BLENotify, 20);

// ========================== DSP FUNCTIONS ==========================

float biquad(float x, float* state, const float* c) {
    float w = x - c[4] * state[0] - c[5] * state[1];
    float y = c[0] * w + c[1] * state[0] + c[2] * state[1];
    state[1] = state[0];
    state[0] = w;
    return y;
}

float filterSample(int ch, float x) {
    for (int b = 0; b < 4; b++)
        x = biquad(x, bpState[ch][b], BP_COEFS[b]);
    x = biquad(x, notchState[ch], NOTCH_COEFS);
    return x;
}

void resetFilterStates() {
    memset(bpState,    0, sizeof(bpState));
    memset(notchState, 0, sizeof(notchState));
}

// ========================== 19-BIT DECOMPRESSION ===================

int32_t extract19bit(const uint8_t* data, int bitOffset) {
    int byteIdx = bitOffset / 8;
    int bitIdx  = bitOffset % 8;

    // Assemble 32 bits starting at byteIdx, then shift/mask to get 19 bits
    uint32_t raw = ((uint32_t)data[byteIdx]     << 24) |
                   ((uint32_t)data[byteIdx + 1] << 16) |
                   ((uint32_t)data[byteIdx + 2] <<  8);
    if (byteIdx + 3 < 19)
        raw |= (uint32_t)data[byteIdx + 3];

    raw <<= bitIdx;
    int32_t val = (int32_t)(raw >> 13) & 0x7FFFF;

    // Sign-extend from 19 bits
    if (val & 0x40000)
        val |= 0xFFF80000;

    return val;
}

void decompressPacket19bit(const uint8_t* pkt) {
    // pkt[0] = packet ID (101-200 for 19-bit compressed)
    // Bits start at pkt[1]; 2 samples x 4 channels x 19 bits = 152 bits
    for (int sample = 0; sample < 2; sample++) {
        for (int ch = 0; ch < NUM_CHANNELS; ch++) {
            int bitOff = (sample * 4 + ch) * 19;
            int32_t delta = extract19bit(&pkt[1], bitOff);
            last_values[ch] += delta;

            float uv = (float)last_values[ch] * GANGLION_UV;

            // Write to PRE-FILTER buffer
            rawBuffer[ch][bufferHead] = uv;
        }

        // Filter each channel and write to POST-FILTER buffer
        for (int ch = 0; ch < NUM_CHANNELS; ch++) {
            float filtered = filterSample(ch, rawBuffer[ch][bufferHead]);
            filteredBuffer[ch][bufferHead] = filtered;
        }

        bufferHead = (bufferHead + 1) % FFT_SIZE;
        sampleCount++;
    }
}

// ========================== DC REMOVAL =============================

void removeDC() {
    for (int ch = 0; ch < NUM_CHANNELS; ch++) {
        float mean = 0.0f;
        for (int i = 0; i < FFT_SIZE; i++)
            mean += rawBuffer[ch][i];
        mean /= FFT_SIZE;
        for (int i = 0; i < FFT_SIZE; i++)
            rawBuffer[ch][i] -= mean;
    }
}

// ========================== RE-FILTER BLOCK ========================

void refilterBlock() {
    resetFilterStates();
    for (int i = 0; i < FFT_SIZE; i++) {
        for (int ch = 0; ch < NUM_CHANNELS; ch++) {
            filteredBuffer[ch][i] = filterSample(ch, rawBuffer[ch][i]);
        }
    }
}

// ========================== BAND POWER =============================

float bandPower(int binLow, int binHigh) {
    float sum = 0.0f;
    for (int i = binLow; i <= binHigh; i++)
        sum += (float)(vReal[i] * vReal[i]);
    return sum;
}

float computeAttention(int ch) {
    for (int i = 0; i < FFT_SIZE; i++) {
        vReal[i] = (double)filteredBuffer[ch][i];
        vImag[i] = 0.0;
    }
    FFT.windowing(FFT_WIN_TYP_HAMMING, FFT_FORWARD);
    FFT.compute(FFT_FORWARD);
    FFT.complexToMagnitude();

    int thetaLow  = (int)round(4.0  / DF);  // bin 5
    int thetaHigh = (int)round(8.0  / DF);  // bin 10
    int alphaLow  = (int)round(8.0  / DF);  // bin 10
    int alphaHigh = (int)round(13.0 / DF);  // bin 17
    int betaLow   = (int)round(13.0 / DF);  // bin 17
    int betaHigh  = (int)round(30.0 / DF);  // bin 38

    float theta = bandPower(thetaLow, thetaHigh);
    float alpha = bandPower(alphaLow, alphaHigh);
    float beta  = bandPower(betaLow,  betaHigh);

    return beta / (alpha + theta + 1e-6f);
}

// ========================== ECG R-PEAK DETECTION ===================

float computeHeartRate() {
    float* ecgData = filteredBuffer[2];  // ch3 = Ganglion eeg3 = index 2

    float maxVal = 0.0f;
    for (int i = 0; i < FFT_SIZE; i++) {
        float absVal = fabs(ecgData[i]);
        if (absVal > maxVal) maxVal = absVal;
    }
    float ecgThreshold = maxVal * 0.6f;

    int peaks = 0;
    int lastPeakIdx = -ECG_REFRACTORY;
    for (int i = 0; i < FFT_SIZE; i++) {
        if (ecgData[i] > ecgThreshold && (i - lastPeakIdx) > ECG_REFRACTORY) {
            peaks++;
            lastPeakIdx = i;
        }
    }

    float durationSec = (float)FFT_SIZE / SAMPLING_FREQ;
    return (float)peaks * (60.0f / durationSec);
}

// ========================== SERIAL PACKET OUTPUT ===================

void sendPacket(float attention, float alpha, float bpm, uint8_t motorState) {
    uint8_t pkt[20];
    pkt[0] = 0xAA;
    pkt[1] = 0x55;
    memcpy(&pkt[2],  &attention, 4);
    memcpy(&pkt[6],  &alpha,     4);
    memcpy(&pkt[10], &bpm,       4);
    pkt[14] = motorState;
    pkt[15] = seqNum++;

    uint16_t cksum = 0;
    for (int i = 2; i < 16; i++)
        cksum += pkt[i];
    memcpy(&pkt[16], &cksum, 2);

    pkt[18] = 0x0D;
    pkt[19] = 0x0A;
    Serial.write(pkt, 20);
    bioChar.writeValue(pkt, 20);
    Serial.println("[TX] Packet sent via Serial + BLE");
}

// ========================== BLOCK PROCESSING =======================

void processBlock() {
    // 1. DC removal on rawBuffer
    removeDC();

    // 2. Re-filter full block into filteredBuffer
    refilterBlock();

    // 3. Attention index: average ch0 (eeg1) and ch1 (eeg2)
    float att0 = computeAttention(0);
    float att1 = computeAttention(1);
    float attention = (att0 + att1) / 2.0f;

    // 4. EMA smoothing
    smoothedAttention = EMA_ALPHA * attention + (1.0f - EMA_ALPHA) * smoothedAttention;

    // 5. Compute alpha power for mental state (re-use last FFT from ch1)
    int alphaLow  = (int)round(8.0  / DF);
    int alphaHigh = (int)round(13.0 / DF);
    float alphaPower = bandPower(alphaLow, alphaHigh);

    // 6. ECG heart rate from filteredBuffer[ch2]
    float bpm = computeHeartRate();

    // 7. Motor control
    uint8_t motorState;
    if (smoothedAttention < ATTENTION_THRESHOLD) {
        digitalWrite(MOTOR_PIN, HIGH);
        motorState = 1;
    } else {
        digitalWrite(MOTOR_PIN, LOW);
        motorState = 0;
    }

    // 8. Send packet
    sendPacket(smoothedAttention, alphaPower, bpm, motorState);

    // 9. Reset
    sampleCount = 0;
}

// ========================== SETUP ==================================

void setup() {
    Serial.begin(115200);
    pinMode(MOTOR_PIN, OUTPUT);
    digitalWrite(MOTOR_PIN, LOW);

    resetFilterStates();
    memset(rawBuffer,      0, sizeof(rawBuffer));
    memset(filteredBuffer,  0, sizeof(filteredBuffer));

    if (!BLE.begin()) {
        while (1);
    }

    BLE.setLocalName("SeeedBioFeedback");
    BLE.setAdvertisedService(bioService);
    bioService.addCharacteristic(bioChar);
    BLE.addService(bioService);
    BLE.advertise();

    BLE.scan();
    BLE.scan();
    Serial.println("[SETUP] BLE init OK, scanning for Ganglion...");
}

// ========================== MAIN LOOP ==============================

void loop() {
    BLEDevice peripheral = BLE.available();
    if (peripheral && peripheral.localName().indexOf("Ganglion") >= 0) {
        Serial.println("[BLE] Found Ganglion, stopping scan...");
        BLE.stopScan();
        if (peripheral.connect()) {
            Serial.println("[BLE] Connected to Ganglion, discovering attributes...");
            peripheral.discoverAttributes();

            BLECharacteristic dataChar =
                peripheral.characteristic("2d30c082-f39f-4ce6-923f-3484ea480596");
            BLECharacteristic commandChar =
                peripheral.characteristic("2d30c083-f39f-4ce6-923f-3484ea480596");

            if (dataChar && commandChar) {
                Serial.println("[BLE] Characteristics found, subscribing...");
                dataChar.subscribe();
                commandChar.writeValue((byte)'b');

                while (peripheral.connected()) {
                    if (dataChar.valueUpdated()) {
                        const uint8_t* pkt = dataChar.value();
                        uint8_t packetId = pkt[0];

                        // 19-bit compressed packets have IDs 101-200
                        if (packetId >= 101 && packetId <= 200) {
                            Serial.print("[DATA] Packet ID=");
                            Serial.print(packetId);
                            Serial.print(" sampleCount=");
                            Serial.println(sampleCount);
                            decompressPacket19bit(pkt);

                            if (sampleCount >= FFT_SIZE) {
                                processBlock();
                            }
                        }
                    }
                }
            } else {
                Serial.println("[BLE] ERROR: characteristics not found!");
            }
            Serial.println("[BLE] Ganglion disconnected, re-scanning...");
            BLE.advertise();
            BLE.scan();
        }
    }
}
