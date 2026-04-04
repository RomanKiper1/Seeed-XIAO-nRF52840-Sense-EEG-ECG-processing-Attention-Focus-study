"""
dashboard_receiver.py
Lightweight Streamlit dashboard that receives pre-computed biofeedback
results from Seeed XIAO nRF52840 via Serial USB or BLE.
No signal processing on PC -- pure visualization.

Usage:
    streamlit run pc/dashboard_receiver.py -- --port COM5
    streamlit run pc/dashboard_receiver.py -- --ble
"""

import argparse
import struct
import sys
import time
from collections import deque

import pandas as pd
import streamlit as st
 
# ========================== PACKET PROTOCOL ==========================
# 28-byte binary packet from Seeed:
#   [0]     0xAA          sync
#   [1]     0x55          sync
#   [2:6]   float32 LE    attention (smoothed)
#   [6:10]  float32 LE    alpha power
#   [10:14] float32 LE    theta power
#   [14:18] float32 LE    beta power
#   [18:22] float32 LE    heart rate BPM
#   [22]    uint8         motor state (0/1)
#   [23]    uint8         sequence number
#   [24:26] uint16 LE     checksum (sum of bytes 2..23)
#   [26]    0x0D          CR
#   [27]    0x0A          LF

SYNC_BYTE_1 = 0xAA
SYNC_BYTE_2 = 0x55
PACKET_SIZE = 28


def verify_checksum(pkt: bytes) -> bool:
    expected = sum(pkt[2:24]) & 0xFFFF
    received = struct.unpack_from('<H', pkt, 24)[0]
    return expected == received


def parse_packet(pkt: bytes):
    """Parse a validated 28-byte packet. Returns (attention, alpha, theta, beta, bpm, motor, seq)."""
    attention = struct.unpack_from('<f', pkt, 2)[0]
    alpha     = struct.unpack_from('<f', pkt, 6)[0]
    theta     = struct.unpack_from('<f', pkt, 10)[0]
    beta      = struct.unpack_from('<f', pkt, 14)[0]
    bpm       = struct.unpack_from('<f', pkt, 18)[0]
    motor     = pkt[22]
    seq       = pkt[23]
    return attention, alpha, theta, beta, bpm, motor, seq


# ========================== SERIAL RECEIVER ==========================

def read_packet_serial(ser):
    """
    Block until a valid 28-byte packet is received from the serial port.
    Implements sync-byte hunting to re-align after corruption.
    """
    while True:
        b = ser.read(1)
        if len(b) == 0:
            continue
        if b[0] != SYNC_BYTE_1:
            continue
        b2 = ser.read(1)
        if len(b2) == 0:
            continue
        if b2[0] != SYNC_BYTE_2:
            continue
        rest = ser.read(26)
        if len(rest) < 26:
            continue
        pkt = bytes([SYNC_BYTE_1, SYNC_BYTE_2]) + rest
        if not verify_checksum(pkt):
            continue
        return parse_packet(pkt)


# ========================== BLE TRANSPORT =============================

class BleTransport:
    """Discovers Seeed XIAO by advertised name and streams packets over BLE."""

    SERVICE_UUID = "19b10000-e8f2-537e-4f6c-d104768a1214"
    CHAR_UUID    = "19b10001-e8f2-537e-4f6c-d104768a1214"
    DEVICE_NAME  = "SeeedBioFeedback"

    def __init__(self, status_callback=None):
        self._status = status_callback or (lambda msg: None)

    async def discover(self):
        """Scan for Seeed by advertised name (most reliable on Windows)."""
        from bleak import BleakScanner
        self._status(f"Scanning for '{self.DEVICE_NAME}'...")
        device = await BleakScanner.find_device_by_name(
            self.DEVICE_NAME, timeout=10.0
        )
        if device is None:
            raise RuntimeError(
                f"Device '{self.DEVICE_NAME}' not found. "
                "Make sure the Seeed XIAO is powered and advertising."
            )
        self._status(f"Found {device.name} ({device.address})")
        return device

    async def stream(self, on_packet):
        """Connect, subscribe, deliver packets via on_packet. Reconnects on drop."""
        import asyncio
        from bleak import BleakClient

        while True:
            try:
                device = await self.discover()
                async with BleakClient(device, timeout=15.0) as client:
                    self._status(f"Connected to {device.name} ({device.address})")
                    queue = asyncio.Queue()

                    def _handler(_, data: bytearray):
                        if (len(data) == PACKET_SIZE
                                and data[0] == SYNC_BYTE_1
                                and data[1] == SYNC_BYTE_2):
                            pkt = bytes(data)
                            if verify_checksum(pkt):
                                queue.put_nowait(parse_packet(pkt))

                    await client.start_notify(self.CHAR_UUID, _handler)

                    while client.is_connected:
                        result = await asyncio.wait_for(
                            queue.get(), timeout=10.0
                        )
                        on_packet(result)

            except asyncio.TimeoutError:
                self._status("Packet timeout — reconnecting...")
            except Exception as exc:
                self._status(f"BLE error: {exc} — reconnecting in 2 s...")
                await asyncio.sleep(2.0)


# ========================== CLI ARGS ================================

def parse_args():
    parser = argparse.ArgumentParser(description="Seeed Biofeedback Dashboard")
    parser.add_argument("--port", type=str, default="COM5",
                        help="Serial port for Seeed XIAO (default: COM5)")
    parser.add_argument("--baud", type=int, default=115200,
                        help="Serial baud rate (default: 115200)")
    parser.add_argument("--ble", action="store_true",
                        help="Use BLE instead of Serial (auto-discovers by name)")
    return parser.parse_known_args()[0]


# ========================== STREAMLIT UI ============================

st.set_page_config(
    page_title="Neuro & Cardio Biofeedback",
    layout="wide",
    page_icon="brain"
)

st.title("Neuro & Cardio Biofeedback Dashboard")
st.markdown("Real-time visualization of **EEG attention** and **ECG heart rate** "
            "processed on Seeed XIAO nRF52840.")

# KPI row
kpi1, kpi2, kpi3, kpi4 = st.columns(4)
with kpi1:
    st_attention = st.empty()
with kpi2:
    st_state = st.empty()
with kpi3:
    st_heart = st.empty()
with kpi4:
    st_motor = st.empty()

# Band power KPI row
bp1, bp2, bp3 = st.columns(3)
with bp1:
    st_theta = st.empty()
with bp2:
    st_alpha = st.empty()
with bp3:
    st_beta = st.empty()

# Chart row
col_left, col_right = st.columns(2)
with col_left:
    st.subheader("Attention Index")
    chart_attention = st.empty()
with col_right:
    st.subheader("Heart Rate (BPM)")
    chart_heart = st.empty()

# Band power chart
st.subheader("Band Powers (Theta / Alpha / Beta)")
chart_bands = st.empty()

# Status bar
status_bar = st.empty()

# Sidebar for connection settings
with st.sidebar:
    st.header("Connection")
    use_ble = st.checkbox("Use BLE", value=False)
    if use_ble:
        st.caption(f"Auto-discovers **{BleTransport.DEVICE_NAME}** by name")
    else:
        serial_port = st.text_input("Serial Port", value="COM5")
        baud_rate = st.number_input("Baud Rate", value=115200, step=9600)
    start_btn = st.button("START", type="primary")

# ========================== MAIN LOOP ===============================

HISTORY_LEN = 120  # ~2.5 min at 1 packet per 1.28 s

def update_ui(attention, alpha, theta, beta, bpm, motor,
              attention_history, heart_history,
              theta_history, alpha_history, beta_history):
    """Push one packet's worth of data into all dashboard widgets."""
    attention_history.append(attention)
    heart_history.append(bpm)
    theta_history.append(theta)
    alpha_history.append(alpha)
    beta_history.append(beta)

    st_attention.metric("ATTENTION INDEX", f"{attention:.2f}",
                        "Focused" if attention > 0.5 else "Distracted")
    mental = "RELAXED" if alpha > 5 else "ACTIVE"
    st_state.metric("MENTAL STATE", mental, f"Alpha: {alpha:.1f}")
    st_heart.metric("HEART RATE", f"{int(bpm)} BPM")
    if motor:
        st_motor.error("MOTOR ACTIVE")
    else:
        st_motor.success("MOTOR OFF")

    prev_theta = theta_history[-2] if len(theta_history) > 1 else theta
    prev_alpha = alpha_history[-2] if len(alpha_history) > 1 else alpha
    prev_beta  = beta_history[-2]  if len(beta_history) > 1  else beta
    st_theta.metric("THETA", f"{theta:.1f}", f"{theta - prev_theta:+.1f}")
    st_alpha.metric("ALPHA", f"{alpha:.1f}", f"{alpha - prev_alpha:+.1f}")
    st_beta.metric("BETA",  f"{beta:.1f}",  f"{beta - prev_beta:+.1f}")

    chart_attention.line_chart(list(attention_history))
    chart_heart.line_chart(list(heart_history))

    band_df = pd.DataFrame({
        "Theta": list(theta_history),
        "Alpha": list(alpha_history),
        "Beta":  list(beta_history),
    })
    chart_bands.line_chart(band_df)


if start_btn:
    attention_history = deque(maxlen=HISTORY_LEN)
    heart_history     = deque(maxlen=HISTORY_LEN)
    theta_history     = deque(maxlen=HISTORY_LEN)
    alpha_history     = deque(maxlen=HISTORY_LEN)
    beta_history      = deque(maxlen=HISTORY_LEN)
    packet_count      = 0

    if use_ble:
        try:
            import asyncio

            def on_packet(result):
                global packet_count
                attention, alpha, theta, beta, bpm, motor, seq = result
                packet_count += 1
                update_ui(attention, alpha, theta, beta, bpm, motor,
                          attention_history, heart_history,
                          theta_history, alpha_history, beta_history)

            transport = BleTransport(
                status_callback=lambda msg: status_bar.info(msg)
            )
            asyncio.run(transport.stream(on_packet))

        except ImportError:
            st.error("BLE mode requires the `bleak` package. Install with: pip install bleak")
        except Exception as e:
            st.error(f"BLE error: {e}")

    else:
        try:
            import serial
            ser = serial.Serial(serial_port, int(baud_rate), timeout=2.0)
            status_bar.success(f"Connected to {serial_port} at {int(baud_rate)} baud")

            while True:
                result = read_packet_serial(ser)
                if result is None:
                    continue
                attention, alpha, theta, beta, bpm, motor, seq = result
                update_ui(attention, alpha, theta, beta, bpm, motor,
                          attention_history, heart_history,
                          theta_history, alpha_history, beta_history)

        except ImportError:
            st.error("Serial mode requires `pyserial`. Install with: pip install pyserial")
        except Exception as e:
            st.error(f"Serial error: {e}")
