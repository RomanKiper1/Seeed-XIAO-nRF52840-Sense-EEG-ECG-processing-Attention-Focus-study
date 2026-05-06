"""List BLE services and characteristics of SeeedBioFeedback (or any device by name)."""
import asyncio
import sys
from bleak import BleakScanner, BleakClient

NAME = sys.argv[1] if len(sys.argv) > 1 else "SeeedBioFeedback"


async def main():
    print(f"Scanning for '{NAME}'...")
    dev = await BleakScanner.find_device_by_name(NAME, timeout=10.0)
    if dev is None:
        print(f"Device '{NAME}' not found.")
        return
    print(f"Found {dev.name}  addr={dev.address}")

    async with BleakClient(dev, timeout=15.0) as client:
        print(f"Connected. Services:\n")
        for s in client.services:
            print(f"Service  {s.uuid}   {s.description}")
            for ch in s.characteristics:
                props = ",".join(ch.properties)
                print(f"  Char   {ch.uuid}   [{props}]")
                for d in ch.descriptors:
                    print(f"     Desc {d.uuid}")
            print()


if __name__ == "__main__":
    asyncio.run(main())
