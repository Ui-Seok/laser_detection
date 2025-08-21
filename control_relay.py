import asyncio
from bleak import BleakClient
import sys

XAIO_ADDRESS = "EA:BD:20:3A:DD:9A"

RELAY_SERVICE_UUID = "19B10000-E8F2-537E-4F6C-D104768A1214"
RELAY_CHARACTERISTIC_UUID = "19B10001-E8F2-537E-4F6C-D104768A1214"

async def main(address):
    print(f"Attempting to connect to {address}")
    async with BleakClient(address) as client:
        if client.is_connected:
            print(f"Successfully connected to {address}")

            while True:
                command = input("Input command: ").strip()

                if command == "1":
                    print("Sending ON command ...")
                    await client.write_gatt_char(RELAY_CHARACTERISTIC_UUID, b'\x01')
                elif command == "0":
                    print("Sending OFF command ...")
                    await client.write_gatt_char(RELAY_CHARACTERISTIC_UUID, b'\x00')
                elif command == "q":
                    print("Existing ...")
                    break
                else:
                    print("Invalid command.")

        else:
            print(f"Failed to connect to {address}")

if __name__ == "__main__":
    try:
        asyncio.run(main(XAIO_ADDRESS))
    except Exception as e:
        print(f"An error occurred: {e}")