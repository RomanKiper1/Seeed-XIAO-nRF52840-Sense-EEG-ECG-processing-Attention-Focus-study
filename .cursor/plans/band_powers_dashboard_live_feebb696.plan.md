---
name: Band powers dashboard live
overview: Expand the MCU packet from 20 to 28 bytes to include theta and beta power alongside alpha, then add real-time band power charts and KPI metrics to the Streamlit dashboard.
todos:
  - id: mcu-packet-expand
    content: Expand BLE characteristic to 28 bytes; compute theta+beta in processBlock; update sendPacket to 28-byte format with theta+beta fields
    status: completed
  - id: dashboard-parser
    content: Update PACKET_SIZE=28, verify_checksum range, parse_packet to extract 7 fields, read_packet_serial to read 26 bytes
    status: completed
  - id: dashboard-ui-bands
    content: Add theta/alpha/beta KPI row + band power chart + wire both BLE and Serial loops to new fields
    status: completed
isProject: false
---

# Live band powers on dashboard

## Problem

The MCU computes theta, alpha, and beta inside `computeAttention()` ([blank_motor_stop.ino](blank_motor_stop/blank_motor_stop.ino) lines 205–207), but only alpha power is sent in the packet (line 281, 297). Theta and beta are discarded.

## New 28-byte packet format

```
[0]     0xAA           sync
[1]     0x55           sync
[2:6]   float32 LE     attention (smoothed)
[6:10]  float32 LE     alpha power
[10:14] float32 LE     theta power    <- NEW
[14:18] float32 LE     beta power     <- NEW
[18:22] float32 LE     bpm
[22]    uint8          motor state
[23]    uint8          sequence number
[24:26] uint16 LE      checksum (sum of bytes 2..23)
[26]    0x0D           CR
[27]    0x0A           LF
```

BLE MTU: nRF52840 supports up to 247 bytes; bleak auto-negotiates. 28 bytes fits comfortably.

---

## Part 1: MCU firmware ([blank_motor_stop.ino](blank_motor_stop/blank_motor_stop.ino))

### 1a. BLE characteristic size (line 83)

Change `20` to `28`:

```c
BLECharacteristic bioChar("19B10001-...", BLERead | BLENotify, 28);
```

### 1b. Compute theta + beta in processBlock() (after line 281)

After existing `alphaPower = bandPower(...)`, add (vReal still holds ch1 FFT state):

```c
int thetaLow  = (int)round(4.0  / DF);
int thetaHigh = (int)round(8.0  / DF);
float thetaPower = bandPower(thetaLow, thetaHigh);

int betaLow  = (int)round(13.0 / DF);
int betaHigh = (int)round(30.0 / DF);
float betaPower = bandPower(betaLow, betaHigh);
```

### 1c. Update sendPacket signature and body

New signature: `sendPacket(float attention, float alpha, float theta, float beta, float bpm, uint8_t motorState)`

Packet assembly (28-byte array):

- bytes 0–1: sync
- bytes 2–5: attention
- bytes 6–9: alpha
- bytes 10–13: theta
- bytes 14–17: beta
- bytes 18–21: bpm
- byte 22: motorState
- byte 23: seqNum++
- bytes 24–25: checksum of bytes 2..23
- bytes 26–27: CR LF

### 1d. Update sendPacket call in processBlock

```c
sendPacket(smoothedAttention, alphaPower, thetaPower, betaPower, bpm, motorState);
```

---

## Part 2: Dashboard ([dashboard_receiver.py](pc/dashboard_receiver.py))

### 2a. Update packet protocol constants and parser

`PACKET_SIZE = 28`. Update `verify_checksum` range: `sum(pkt[2:24])`, checksum at offset 24. Update `parse_packet` to extract 7 fields:

```python
def parse_packet(pkt: bytes):
    attention = struct.unpack_from('<f', pkt, 2)[0]
    alpha     = struct.unpack_from('<f', pkt, 6)[0]
    theta     = struct.unpack_from('<f', pkt, 10)[0]
    beta      = struct.unpack_from('<f', pkt, 14)[0]
    bpm       = struct.unpack_from('<f', pkt, 18)[0]
    motor     = pkt[22]
    seq       = pkt[23]
    return attention, alpha, theta, beta, bpm, motor, seq
```

### 2b. Update Serial reader

`read_packet_serial`: read 26 remaining bytes after sync (not 18).

### 2c. Add UI elements for band powers

New KPI row below existing one with 3 columns for Theta, Alpha, Beta — each with `st.metric()` showing the numeric value and delta from previous.

New chart row: a single `st.line_chart` with 3 series (theta, alpha, beta histories) so they overlay on one plot, or 3 stacked charts.

### 2d. Wire packet data in both BLE and Serial loops

Update `on_packet` callback and serial loop to unpack 7 fields, append to `theta_history`, `alpha_history`, `beta_history` deques, and update the new metrics + charts.

---

## Output on dashboard

- **KPI tiles:** `THETA` / `ALPHA` / `BETA` with raw power values (like the screenshot shows `Alpha: 570677.5`)
- **Band powers chart:** 3 colored lines (theta, alpha, beta) over time
- Existing attention, mental state, heart rate, and motor KPIs remain unchanged
