#!/usr/bin/env python3
"""
UDP-based video streaming server for Project Sauron
Streams MJPEG video over UDP with WebSocket tunnel for browser compatibility
Includes thermal camera control (brightness, contrast, palette, etc.)
"""

import socket
import threading
import time
import struct
import json
import asyncio
import websockets
from typing import Dict, Optional, Set
from collections import deque
import cv2
import serial
from pathlib import Path

# Camera imports
import sys
sys.path.insert(0, str(Path(__file__).parent))
from camapi import FrameGrabber, list_cameras

# ============================================================================
# Protocol Constants (Same as sliderserver.py)
# ============================================================================
BEGIN_BYTE = 0xF0
END_BYTE = 0xFF
DEVICE_ADDR = 0x36
CLASS_ADDR_IMAGE = 0x70
SUBCLASS_ADDR_BRIGHTNESS = 0x02
SUBCLASS_ADDR_CONTRAST = 0x03
SUBCLASS_ADDR_DETAIL_ENHANCE = 0x10
RW_WRITE = 0x00
RW_READ = 0x01

# New protocol addresses (from HM-TM5X protocol guide)
CLASS_ADDR_AUTO_SHUTTER = 0x7C  # Auto shutter control
SUBCLASS_ADDR_AUTO_SHUTTER = 0x04  # Values 0-3

CLASS_ADDR_DENOISE = 0x78  # Denoise control
SUBCLASS_ADDR_STATIC_DENOISE = 0x15  # Range 0-100
SUBCLASS_ADDR_DYNAMIC_DENOISE = 0x16  # Range 0-100

CLASS_ADDR_SETTINGS = 0x74  # Settings control
SUBCLASS_ADDR_FACTORY_RESET = 0x0F  # Write only
SUBCLASS_ADDR_SAVE_SETTINGS = 0x10  # Write only
SUBCLASS_ADDR_FLIP = 0x04
RW_WRITE = 0x00
RW_READ = 0x01

# UDP Settings
UDP_PORT = 9000
WEBSOCKET_PORT = 8081
MAX_PACKET_SIZE = 1472  # 1500 MTU - 28 bytes (IP + UDP headers)
FRAME_TIMEOUT = 5  # seconds

# Global FPS control (can be modified via commands)
current_fps = 10.0  # Default FPS

# ============================================================================
# Serial Link (Same as sliderserver.py)
# ============================================================================
class SerialLink:
    def __init__(self, port: str):
        self.port = port
        self.ser = None
        self.lock = threading.Lock()
        self._connect()

    def _connect(self):
        try:
            self.ser = serial.Serial(self.port, baudrate=115200, timeout=1.0)
            time.sleep(0.5)
            self.ser.reset_input_buffer()
            self.ser.reset_output_buffer()
            self.ser.dtr = False
            self.ser.rts = False
            print(f"[Serial] Connected to {self.port}")
        except Exception as e:
            print(f"[Serial] Failed to connect to {self.port}: {e}")
            self.ser = None

    def send_recv(self, data: bytes, retries: int = 2) -> bytes:
        if not self.ser:
            return b""
        
        with self.lock:
            for attempt in range(retries):
                try:
                    self.ser.write(data)
                    response = b""
                    timeout_count = 0
                    while timeout_count < 10:
                        chunk = self.ser.read(256)
                        if not chunk:
                            timeout_count += 1
                            time.sleep(0.01)
                        else:
                            response += chunk
                            timeout_count = 0
                            if response and response[-1] == END_BYTE:
                                return response
                    if response:
                        return response
                except Exception as e:
                    print(f"[Serial] Attempt {attempt + 1} failed: {e}")
                    if attempt < retries - 1:
                        time.sleep(0.1)
        return b""

# ============================================================================
# Thermal Control Functions
# ============================================================================
def _build_packet(class_addr: int, subclass_addr: int, rw_flag: int, data: bytes) -> bytes:
    """Build HM-TM5X protocol packet"""
    size = 1 + 1 + 1 + len(data)
    packet = bytearray([BEGIN_BYTE, size, DEVICE_ADDR, class_addr, subclass_addr, rw_flag])
    packet.extend(data)
    checksum = (DEVICE_ADDR + class_addr + subclass_addr + rw_flag + sum(data)) & 0xFF
    packet.append(checksum)
    packet.append(END_BYTE)
    return bytes(packet)

def set_u8_setting(class_addr: int, subclass_addr: int, value: int, serial_link: SerialLink) -> Dict:
    """Set a 0-255 value on thermal camera"""
    value = max(0, min(255, int(value)))
    packet = _build_packet(class_addr, subclass_addr, RW_WRITE, bytes([value]))
    response = serial_link.send_recv(packet)
    return {"success": bool(response), "value": value}

def set_brightness(value: int, serial_link: SerialLink) -> Dict:
    return set_u8_setting(CLASS_ADDR_IMAGE, SUBCLASS_ADDR_BRIGHTNESS, value, serial_link)

def set_contrast(value: int, serial_link: SerialLink) -> Dict:
    return set_u8_setting(CLASS_ADDR_IMAGE, SUBCLASS_ADDR_CONTRAST, value, serial_link)

def set_detail_enhance(value: int, serial_link: SerialLink) -> Dict:
    return set_u8_setting(CLASS_ADDR_IMAGE, SUBCLASS_ADDR_DETAIL_ENHANCE, value, serial_link)

def set_flip(mode: int, serial_link: SerialLink) -> Dict:
    return set_u8_setting(CLASS_ADDR_IMAGE, SUBCLASS_ADDR_FLIP, mode, serial_link)

def set_auto_shutter(mode: int, serial_link: SerialLink) -> Dict:
    # 0-3: different auto shutter modes
    mode = int(max(0, min(3, mode)))
    return set_u8_setting(CLASS_ADDR_AUTO_SHUTTER, SUBCLASS_ADDR_AUTO_SHUTTER, mode, serial_link)

def set_static_denoise(value: int, serial_link: SerialLink) -> Dict:
    # 0-100 range
    value = int(max(0, min(100, value)))
    return set_u8_setting(CLASS_ADDR_DENOISE, SUBCLASS_ADDR_STATIC_DENOISE, value, serial_link)

def set_dynamic_denoise(value: int, serial_link: SerialLink) -> Dict:
    # 0-100 range
    value = int(max(0, min(100, value)))
    return set_u8_setting(CLASS_ADDR_DENOISE, SUBCLASS_ADDR_DYNAMIC_DENOISE, value, serial_link)

def factory_reset(serial_link: SerialLink) -> Dict:
    # Write-only command, no data
    return set_u8_setting(CLASS_ADDR_SETTINGS, SUBCLASS_ADDR_FACTORY_RESET, 0, serial_link)

def save_settings(serial_link: SerialLink) -> Dict:
    # Write-only command, no data
    return set_u8_setting(CLASS_ADDR_SETTINGS, SUBCLASS_ADDR_SAVE_SETTINGS, 0, serial_link)

# ============================================================================
# UDP Video Streamer
# ============================================================================
class UDPVideoStreamer:
    def __init__(self, camera_id: str, quality: int = 80):
        self.camera_id = camera_id
        self.quality = quality
        self.frame_grabber = FrameGrabber(camera_id)
        self.current_frame = None
        self.frame_lock = threading.Lock()
        self.running = False
        self.frame_sequence = 0
        self.clients: Set[tuple] = set()
        self.client_lock = threading.Lock()

    def start_capture(self):
        if not self.running:
            self.running = True
            self.frame_grabber.start()
            threading.Thread(target=self._capture_loop, daemon=True).start()

    def stop_capture(self):
        self.running = False
        self.frame_grabber.stop()

    def _capture_loop(self):
        """Continuously capture frames"""
        while self.running:
            try:
                frame = self.frame_grabber.get_frame()
                if frame is not None:
                    with self.frame_lock:
                        self.current_frame = frame
            except Exception as e:
                print(f"[Capture] Error: {e}")
                time.sleep(0.1)

    def register_client(self, addr: tuple):
        with self.client_lock:
            self.clients.add(addr)

    def unregister_client(self, addr: tuple):
        with self.client_lock:
            self.clients.discard(addr)

    def get_clients(self) -> Set[tuple]:
        with self.client_lock:
            return self.clients.copy()

    def encode_frame(self) -> Optional[bytes]:
        """Encode current frame as MJPEG"""
        with self.frame_lock:
            if self.current_frame is None:
                return None
            _, jpeg = cv2.imencode('.jpg', self.current_frame, 
                                  [cv2.IMWRITE_JPEG_QUALITY, self.quality])
            return jpeg.tobytes()

    def fragment_frame(self, frame_data: bytes) -> list:
        """Fragment frame into UDP packets with headers"""
        packets = []
        frame_id = self.frame_sequence
        self.frame_sequence += 1
        
        total_chunks = (len(frame_data) + MAX_PACKET_SIZE - 1) // MAX_PACKET_SIZE
        
        for chunk_idx in range(total_chunks):
            start = chunk_idx * MAX_PACKET_SIZE
            end = min(start + MAX_PACKET_SIZE, len(frame_data))
            chunk = frame_data[start:end]
            
            # Packet format: [FRAME_ID(4)][CHUNK_IDX(2)][TOTAL_CHUNKS(2)][DATA]
            header = struct.pack('>IHH', frame_id, chunk_idx, total_chunks)
            packet = header + chunk
            packets.append(packet)
        
        return packets

    def stream_loop(self, sock: socket.socket):
        """Send frames to all registered clients"""
        last_frame_time = 0
        while self.running:
            try:
                global current_fps
                current_time = time.time()
                frame_period = 1.0 / max(1.0, current_fps)  # Dynamic FPS control
                if current_time - last_frame_time < frame_period:
                    time.sleep(0.01)
                    continue
                
                frame_data = self.encode_frame()
                if not frame_data:
                    time.sleep(0.01)
                    continue
                
                packets = self.fragment_frame(frame_data)
                clients = self.get_clients()
                
                for packet in packets:
                    for client_addr in clients:
                        try:
                            sock.sendto(packet, client_addr)
                        except Exception as e:
                            print(f"[UDP] Send error to {client_addr}: {e}")
                
                last_frame_time = current_time
            except Exception as e:
                print(f"[Stream] Error: {e}")
                time.sleep(0.1)

# ============================================================================
# UDP Server
# ============================================================================
class UDPServer:
    def __init__(self, port: int = UDP_PORT):
        self.port = port
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.sock.bind(('0.0.0.0', port))
        self.running = False
        self.streamers: Dict[str, UDPVideoStreamer] = {}
        self.serial_port = self._find_serial_port()
        self.serial_link = SerialLink(self.serial_port) if self.serial_port else None

    def _find_serial_port(self) -> Optional[str]:
        """Find thermal camera serial port"""
        import glob
        ports = glob.glob('/dev/serial/by-id/usb-Prolific*')
        if ports:
            return ports[0]
        ports = glob.glob('/dev/ttyUSB*')
        if ports:
            return ports[0]
        return None

    def add_streamer(self, camera_id: str, quality: int = 80):
        if camera_id not in self.streamers:
            self.streamers[camera_id] = UDPVideoStreamer(camera_id, quality)
            print(f"[UDP] Added streamer for {camera_id}")

    def start(self):
        self.running = True
        print(f"[UDP] Server listening on port {self.port}")
        
        # Start stream threads
        for streamer in self.streamers.values():
            streamer.start_capture()
            threading.Thread(target=streamer.stream_loop, args=(self.sock,), daemon=True).start()
        
        # Start server loop
        threading.Thread(target=self._server_loop, daemon=True).start()

    def _server_loop(self):
        """Handle client registrations and commands"""
        while self.running:
            try:
                data, addr = self.sock.recvfrom(512)
                message = json.loads(data.decode())
                self._handle_command(message, addr)
            except socket.timeout:
                continue
            except Exception as e:
                print(f"[UDP] Error: {e}")

    def _handle_command(self, msg: dict, addr: tuple):
        """Handle incoming commands"""
        cmd = msg.get("cmd")
        
        if cmd == "register":
            camera_id = msg.get("camera")
            if camera_id in self.streamers:
                self.streamers[camera_id].register_client(addr)
                print(f"[UDP] Client {addr} registered for {camera_id}")
        
        elif cmd == "brightness":
            if self.serial_link:
                result = set_brightness(msg.get("value", 128), self.serial_link)
                print(f"[Thermal] Brightness: {result}")
        
        elif cmd == "contrast":
            if self.serial_link:
                result = set_contrast(msg.get("value", 128), self.serial_link)
                print(f"[Thermal] Contrast: {result}")
        
        elif cmd == "detail":
            if self.serial_link:
                result = set_detail_enhance(msg.get("value", 128), self.serial_link)
                print(f"[Thermal] Detail: {result}")
        
        elif cmd == "flip":
            if self.serial_link:
                result = set_flip(msg.get("value", 0), self.serial_link)
                print(f"[Thermal] Flip: {result}")
        
        elif cmd == "auto-shutter":
            if self.serial_link:
                result = set_auto_shutter(msg.get("value", 0), self.serial_link)
                print(f"[Thermal] Auto Shutter: {result}")
        
        elif cmd == "static-denoise":
            if self.serial_link:
                result = set_static_denoise(msg.get("value", 50), self.serial_link)
                print(f"[Thermal] Static Denoise: {result}")
        
        elif cmd == "dynamic-denoise":
            if self.serial_link:
                result = set_dynamic_denoise(msg.get("value", 50), self.serial_link)
                print(f"[Thermal] Dynamic Denoise: {result}")
        
        elif cmd == "factory-reset":
            if self.serial_link:
                result = factory_reset(self.serial_link)
                print(f"[Thermal] Factory Reset: {result}")
        
        elif cmd == "save-settings":
            if self.serial_link:
                result = save_settings(self.serial_link)
                print(f"[Thermal] Save Settings: {result}")
        
        elif cmd == "fps":
            global current_fps
            fps = msg.get("value", 10)
            current_fps = max(1.0, min(60.0, float(fps)))
            print(f"[Stream] FPS changed to {current_fps}")

    def stop(self):
        self.running = False
        for streamer in self.streamers.values():
            streamer.stop_capture()
        self.sock.close()

# ============================================================================
# WebSocket Tunnel (for browser support)
# ============================================================================
class WebSocketTunnel:
    def __init__(self, udp_server: UDPServer, port: int = WEBSOCKET_PORT):
        self.udp_server = udp_server
        self.port = port
        self.clients: Set[websockets.WebSocketServerProtocol] = set()

    async def handle_client(self, websocket, path):
        """Handle WebSocket client connection"""
        self.clients.add(websocket)
        addr = (websocket.remote_address[0], 9001)  # Virtual UDP addr
        
        try:
            async for message in websocket:
                # Forward commands to UDP server
                msg = json.loads(message)
                self.udp_server._handle_command(msg, addr)
                
                # Register for video if requested
                if msg.get("cmd") == "register":
                    camera_id = msg.get("camera")
                    if camera_id in self.udp_server.streamers:
                        streamer = self.udp_server.streamers[camera_id]
                        # Send frames via WebSocket
                        asyncio.create_task(self._stream_video(websocket, streamer))
        except Exception as e:
            print(f"[WebSocket] Error: {e}")
        finally:
            self.clients.discard(websocket)

    async def _stream_video(self, websocket, streamer: UDPVideoStreamer):
        """Stream video frames via WebSocket"""
        try:
            while websocket in self.clients:
                frame_data = streamer.encode_frame()
                if frame_data:
                    await websocket.send(frame_data)
                await asyncio.sleep(0.033)  # ~30 FPS
        except Exception as e:
            print(f"[Video Stream] Error: {e}")

    async def start(self):
        """Start WebSocket server"""
        print(f"[WebSocket] Server listening on port {self.port}")
        async with websockets.serve(self.handle_client, "0.0.0.0", self.port):
            await asyncio.Future()  # Run forever

# ============================================================================
# Main
# ============================================================================
if __name__ == "__main__":
    # Initialize UDP server
    udp_server = UDPServer(port=UDP_PORT)
    
    # Add cameras
    try:
        cameras = list_cameras()
        if len(cameras) > 0:
            udp_server.add_streamer(cameras[0][0])  # RGB
        if len(cameras) > 1:
            udp_server.add_streamer(cameras[1][0])  # Thermal
    except Exception as e:
        print(f"[Main] Camera detection failed: {e}")
    
    udp_server.start()
    
    # Start WebSocket tunnel
    tunnel = WebSocketTunnel(udp_server)
    asyncio.run(tunnel.start())
