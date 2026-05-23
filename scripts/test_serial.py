import time
import serial
from serial.tools import list_ports

ports = list(list_ports.comports())

if not ports:
    raise RuntimeError("COMポートが見つかりません")

port = ports[0].device
print(f"Using port: {port}")

ser = serial.Serial(port, 115200, timeout=1)
time.sleep(2)

ser.write(b"A\n")
ser.flush()
ser.close()

print("Sent: A")
