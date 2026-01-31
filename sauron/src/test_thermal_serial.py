#-------------------------------------------------------------------------------------------
# AXIS FLYING HM-TM5X-XRG/C SERIES THERMAL CAMERA MODULE
# <link to the website/manual goes here>
#-------------------------------------------------------------------------------------------
# Description:
#   This sample reads data from the thermal camera connected over USB.
#
# Notes:
# 	Requires the pySerial module.
#
# Configuration:
#	'Active data port' must be 'USB distance in m'.
#	'USB port update rate' will determine readings per second.
#
#-------------------------------------------------------------------------------------------

import serial

print('Running sample.')

# Make a connection to the com port. USB0 is the first default port assigned to USB serial devices.
serialPortName = '/dev/ttyUSB0'
serialPortBaudRate = 115200

port = serial.Serial(serialPortName, serialPortBaudRate, timeout=0.1)

def get_CHK(device_address, class_address, subclass_address, rw_flag, data):
    return (device_address+class_address+subclass_address+rw_flag+data) & 0xFF

def print_bytes(ba):
    for i in range(len(ba)):
        print(hex(ba[i]))

N = 1
BEGIN = 0xF0
SIZE = 0x04+N
DEVICE_ADDRESS = 0x36
CLASS_ADDRESS = 0x78
SUBCLASS_ADDRESS = 0x20
FLAG = 0x00
DATA = 0x00
CHK = get_CHK(DEVICE_ADDRESS, CLASS_ADDRESS,SUBCLASS_ADDRESS, FLAG, DATA)
END = 0xFF

packet = bytearray(8+SIZE-4)

packet[0] = BEGIN
packet[1] = SIZE
packet[2] = DEVICE_ADDRESS
packet[3] = CLASS_ADDRESS
packet[4] = SUBCLASS_ADDRESS
packet[5] = FLAG
packet[6] = DATA
packet[7] = CHK
packet[8] = END

print_bytes(packet)
# Clear buffer of any partial responses.
port.readline()

#port.write(b'\xF0\x05\x36\x78\x02\x00\x64\x14\xFF')
port.write(packet)
print("wrote line")
# Continuously gather distance data.
while True:	
    line_read= port.readline()
    if (0 != len(line_read)):
        print(line_read)
        print(type(line_read))
        print_bytes(line_read)
        break


    # Convert the string to a numeric distance value.
    '''
    try:
        splitStr = distanceStr.split(b" ")
        distance = int(splitStr[0])
    except ValueError:
        # It is possible that the SF30 does not get a valid signal, we represent this case as a -1.0m.
        distance = -1.0
        
    # Do what you want with the distance information here.
    print(distance)
    '''
