"""Subscribe to SeeedBioFeedback's notify characteristic and dump raw bytes.
Pure diagnostic — no parsing, no checksum, just hex of whatever arrives."""
import asyncio
from bleak import BleakScanner, BleakClient

DEVICE_NAME = "SeeedBioFeedback"
CHAR_UUID   = "19b10001-e8f2-537e-4f6c-d104768a1214"


async def main():
    print(f"Scanning for '{DEVICE_NAME}'...")
    dev = await BleakScanner.find_device_by_name(DEVICE_NAME, timeout=10.0)
    if dev is None:
        print("Device not found.")
        return
    print(f"Found {dev.name}  addr={dev.address}")

    async with BleakClient(dev, timeout=15.0) as client:
        print(f"Connected. MTU = {client.mtu_size} (payload max ~ MTU-3)")

        count = 0

        def handler(_, data: bytearray):
            nonlocal count
            count += 1
            head = data[:2].hex()
            tail = data[-2:].hex()
            print(f"[#{count:04d}] len={len(data):2d}  "
                  f"head={head}  tail={tail}  hex={data.hex()}")

        await client.start_notify(CHAR_UUID, handler)
        print("Subscribed. Waiting 30 s for notifications...")
        await asyncio.sleep(30.0)
        await client.stop_notify(CHAR_UUID)
        print(f"Done. Total notifications: {count}")


if __name__ == "__main__":
    asyncio.run(main())
