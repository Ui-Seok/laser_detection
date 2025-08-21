import asyncio
from bleak import BleakScanner

async def main():
    print("Scanning for devices...")
    devices = await BleakScanner.discover()
    for d in devices:
        if d.name:
            print(f"Device Name: {d.name}, Address: {d.address}")

if __name__ == "__main__":
    asyncio.run(main())