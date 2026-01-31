import os
import time
import threading
import logging
from typing import Dict, Optional, Tuple

import cv2
from flask import Flask, Response, jsonify, render_template_string, request

# Setup logging for diagnostics
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

"""
Adds a second live MJPEG stream for a thermal camera (or any second UVC device).
- Primary RGB stream:   /stream            (defaults to camera index 0)
- Thermal stream:       /thermal           (defaults to camera index 1)
You can override device indices via env:
  CAMERA_INDEX=0 THERMAL_INDEX=1 python3 server.py
This implementation runs ONE capture thread per device and fans out frames,
so starting the thermal stream will not steal frames from the RGB stream (and vice versa).
"""

CAMERA_DEVICE = os.environ.get("CAMERA_DEVICE")
THERMAL_DEVICE = os.environ.get("THERMAL_DEVICE")

def _parse_device(env_value: str | None, default_index_env: str, fallback: int) -> str | int:
    if env_value:
        return env_value  # explicit /dev/videoX path or backend string
    try:
        return int(os.environ.get(default_index_env, fallback))
    except ValueError:
        return fallback

CAMERA_INDEX  = _parse_device(CAMERA_DEVICE, "CAMERA_INDEX", 0)
THERMAL_INDEX = _parse_device(THERMAL_DEVICE, "THERMAL_INDEX", 1)
# RGB camera: 16:9 (optical)
CAM_WIDTH  = int(os.environ.get("CAM_WIDTH",  1280))
CAM_HEIGHT = int(os.environ.get("CAM_HEIGHT", 720))
# Thermal camera: 4:3 (thermal imaging)
THERMAL_WIDTH  = int(os.environ.get("THERMAL_WIDTH",  640))
THERMAL_HEIGHT = int(os.environ.get("THERMAL_HEIGHT", 480))
FPS    = float(os.environ.get("CAM_FPS",   10))  # preview throttle per endpoint

# Global FPS control (can be modified via API)
current_fps = FPS

SERIAL_PORT = os.environ.get(
    "THERMAL_SERIAL_PORT",
    "/dev/serial/by-id/usb-Prolific_Technology_Inc._USB-Serial_Controller-if00-port0",
)
SERIAL_BAUD = int(os.environ.get("THERMAL_SERIAL_BAUD", 115200))
SERIAL_TIMEOUT = float(os.environ.get("THERMAL_SERIAL_TIMEOUT", 0.2))
SERIAL_READ_BYTES = int(os.environ.get("THERMAL_SERIAL_READ_BYTES", 256))

# Palette protocol constants (HM-TM5X UART/CVBS)
BEGIN_BYTE = 0xF0
END_BYTE = 0xFF
DEVICE_ADDR = 0x36
CLASS_ADDR_PALETTE = 0x78
SUBCLASS_ADDR_PALETTE = 0x20
RW_WRITE = 0x00
RW_READ = 0x01

# Other protocol addresses (HM-TM5X UART/CVBS)
CLASS_ADDR_IMAGE = 0x70  # image control (brightness/contrast/detail/flip)
SUBCLASS_ADDR_MIRROR = 0x11

SUBCLASS_ADDR_BRIGHTNESS = 0x02
SUBCLASS_ADDR_CONTRAST = 0x03
SUBCLASS_ADDR_DETAIL_ENHANCE = 0x10

# New protocol addresses (from HM-TM5X protocol guide)
CLASS_ADDR_AUTO_SHUTTER = 0x7C  # Auto shutter control
SUBCLASS_ADDR_AUTO_SHUTTER = 0x04  # Values 0-3

CLASS_ADDR_DENOISE = 0x78  # Denoise control
SUBCLASS_ADDR_STATIC_DENOISE = 0x15  # Range 0-100
SUBCLASS_ADDR_DYNAMIC_DENOISE = 0x16  # Range 0-100

CLASS_ADDR_SETTINGS = 0x74  # Settings control
SUBCLASS_ADDR_FACTORY_RESET = 0x0F  # Write only
SUBCLASS_ADDR_SAVE_SETTINGS = 0x10  # Write only

PALETTES: Dict[str, int] = {
    "white_hot": 0x00,
    "black_hot": 0x01,
    "fusion_1": 0x02,
    "rainbow": 0x03,
    "fusion_2": 0x04,
    "iron_red_1": 0x05,
    "iron_red_2": 0x06,
    "dark_brown": 0x07,
    "color_1": 0x08,
    "color_2": 0x09,
    "ice_fire": 0x0A,
    "rain": 0x0B,
    "green_hot": 0x0C,
    "red_hot": 0x0D,
    "deep_blue": 0x0E,
}

JPEG_QUALITY = int(os.environ.get("JPEG_QUALITY", 60))

app = Flask(__name__)

class FrameGrabber:
    """
    Threaded frame grabber for one V4L2/UVC/AVFoundation device.
    Reads continuously into `latest` and allows multiple consumers to encode MJPEG
    without touching the capture handle from multiple threads.
    """
    def __init__(self, device_index, width=640, height=480, request_mjpg=True):
        self.device_index = device_index  # may be int index or /dev/videoX string
        self.width = width
        self.height = height
        self.request_mjpg = request_mjpg
        self.cap = None
        self.running = False
        self.latest = None
        self.last_ts = 0.0
        self.lock = threading.Lock()
        self.thread = None

    def start(self):
        if self.running:
            return
        backend_flag = cv2.CAP_V4L2 if hasattr(cv2, "CAP_V4L2") else None
        tried_backends = []

        def _try_open(backend=None):
            tried_backends.append(backend)
            return cv2.VideoCapture(self.device_index, backend) if backend is not None else cv2.VideoCapture(self.device_index)

        self.cap = _try_open(backend_flag)
        if not self.cap.isOpened():
            # Fallback to default backend if V4L2 fails
            self.cap = _try_open(None)

        if self.request_mjpg:
            self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        self.cap.set(cv2.CAP_PROP_FPS, FPS)
        if not self.cap.isOpened():
            raise RuntimeError(f"Failed to open video device {self.device_index} (backends tried: {tried_backends})")
        self.running = True
        self.thread = threading.Thread(target=self._loop, daemon=True)
        self.thread.start()

    def _loop(self):
        # Continuous grab; store latest frame
        while self.running and self.cap.isOpened():
            ok, frame = self.cap.read()
            if not ok:
                # brief backoff if device stalls
                time.sleep(0.01)
                continue
            with self.lock:
                self.latest = frame
                self.last_ts = time.time()

    def read_latest(self):
        with self.lock:
            if self.latest is None:
                return None
            return self.latest.copy()

    def latest_shape(self):
        """Return a simple shape dict for the latest frame or None."""
        with self.lock:
            if self.latest is None:
                return None
            shape = getattr(self.latest, 'shape', None)
            if not shape:
                return None
            # shape is (h, w) or (h, w, c)
            h = int(shape[0])
            w = int(shape[1])
            channels = int(shape[2]) if len(shape) >= 3 else 1
            return {"width": w, "height": h, "channels": channels}

    def stop(self):
        self.running = False
        if self.thread is not None:
            self.thread.join(timeout=1.0)
        if self.cap is not None:
            try:
                self.cap.release()
            except:
                pass
        with self.lock:
            self.latest = None
        self.thread = None
        self.cap = None

    def reconnect(self):
        self.stop()
        self.start()

    def status(self) -> Dict[str, object]:
        return {
            "running": bool(self.running),
            "device": self.device_index,
            "last_frame_ts": self.last_ts,
        }

    def set_device(self, device_index, restart: bool = False):
        """Assign a new device path/index; optionally restart capture if it was running."""
        was_running = bool(self.running)
        if self.running:
            self.stop()
        self.device_index = device_index
        if restart or was_running:
            self.start()

# Instantiate grabbers for both cameras
rgb = FrameGrabber(CAMERA_INDEX, CAM_WIDTH, CAM_HEIGHT, request_mjpg=True)
thermal = FrameGrabber(THERMAL_INDEX, THERMAL_WIDTH, THERMAL_HEIGHT, request_mjpg=True)


def _grabber_for(name: str) -> Tuple[str, FrameGrabber]:
    lower = name.lower()
    if lower == "rgb":
        return "rgb", rgb
    if lower == "thermal":
        return "thermal", thermal
    raise ValueError(f"Unknown camera '{name}'")

def swap_camera_devices() -> Dict[str, object]:
    """
    Swap device assignments between RGB and Thermal grabbers, restarting them
    only after both have been cleanly stopped. This avoids overlapping opens
    on the same /dev/video* node.
    """
    rgb_running = rgb.running
    thermal_running = thermal.running
    rgb_dev = rgb.device_index
    thermal_dev = thermal.device_index

    # Stop both first to avoid simultaneous opens on the same node
    if rgb_running:
        rgb.stop()
    if thermal_running:
        thermal.stop()

    # Brief pause to let V4L2 release file handles
    time.sleep(0.2)

    rgb.device_index = thermal_dev
    thermal.device_index = rgb_dev

    try:
        if rgb_running:
            rgb.start()
        if thermal_running:
            thermal.start()
    except Exception as exc:
        # Roll back to previous assignment on failure
        rgb.device_index = rgb_dev
        thermal.device_index = thermal_dev
        # Attempt to restore previous running state
        try:
            if rgb_running and not rgb.running:
                rgb.start()
            if thermal_running and not thermal.running:
                thermal.start()
        finally:
            pass
        raise RuntimeError(f"Swap failed: {exc}")

    return {"rgb": rgb.status(), "thermal": thermal.status()}

def mjpeg_generator(source: FrameGrabber, preview_fps: float):
    period = 1.0 / max(1.0, preview_fps)
    try:
        while True:
            frame = source.read_latest()
            if frame is None:
                time.sleep(0.05)
                continue

            ok, jpg = cv2.imencode(
                '.jpg', frame,
                [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_QUALITY]
            )
            if not ok:
                time.sleep(0.01)
                continue

            yield (
                b'--frame\r\n'
                b'Content-Type: image/jpeg\r\n\r\n' +
                jpg.tobytes() +
                b'\r\n'
            )
            time.sleep(period)
    except GeneratorExit:
        # Client disconnected cleanly
        return
    except Exception as e:
        print(f"[WARN] MJPEG generator error: {e}")
        return

@app.route('/stream')
def stream_rgb():
    global current_fps
    return Response(mjpeg_generator(rgb, current_fps),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/thermal')
def stream_thermal():
    global current_fps
    return Response(mjpeg_generator(thermal, current_fps),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

def _palette_packet(palette_code: int) -> bytes:
    palette_code &= 0xFF
    size = 1 + 4
    checksum = (DEVICE_ADDR + CLASS_ADDR_PALETTE + SUBCLASS_ADDR_PALETTE + RW_WRITE + palette_code) & 0xFF
    pkt = bytearray(9)
    pkt[0] = BEGIN_BYTE
    pkt[1] = size
    pkt[2] = DEVICE_ADDR
    pkt[3] = CLASS_ADDR_PALETTE
    pkt[4] = SUBCLASS_ADDR_PALETTE
    pkt[5] = RW_WRITE
    pkt[6] = palette_code
    pkt[7] = checksum
    pkt[8] = END_BYTE
    return bytes(pkt)

def _build_packet(class_addr: int, subclass_addr: int, rw_flag: int, data: bytes = b"") -> bytes:
    data = data or b""
    size = len(data) + 4
    checksum = (DEVICE_ADDR + (class_addr & 0xFF) + (subclass_addr & 0xFF) + (rw_flag & 0xFF) + sum(data)) & 0xFF
    pkt = bytearray(8 + len(data))
    pkt[0] = BEGIN_BYTE
    pkt[1] = size & 0xFF
    pkt[2] = DEVICE_ADDR
    pkt[3] = class_addr & 0xFF
    pkt[4] = subclass_addr & 0xFF
    pkt[5] = rw_flag & 0xFF
    if data:
        pkt[6:6 + len(data)] = data
        pkt[6 + len(data)] = checksum
        pkt[7 + len(data)] = END_BYTE
    else:
        pkt[6] = checksum
        pkt[7] = END_BYTE
    return bytes(pkt)


def _parse_reply(resp: bytes) -> Dict[str, object]:
    """Best-effort parser for HM-TM5X reply packets."""
    if not resp:
        return {"ok": False, "error": "empty response"}
    try:
        start = resp.index(bytes([BEGIN_BYTE]))
    except ValueError:
        return {"ok": False, "error": "no BEGIN byte", "raw": _pretty_bytes(resp)}

    if start + 8 > len(resp):
        return {"ok": False, "error": "short response", "raw": _pretty_bytes(resp)}

    size = resp[start + 1]
    dev = resp[start + 2]
    class_addr = resp[start + 3]
    subclass_addr = resp[start + 4]
    rw = resp[start + 5]

    data_len = max(0, size - 4)
    data_start = start + 6
    data_end = data_start + data_len
    if data_end + 2 > len(resp):
        return {"ok": False, "error": "incomplete payload", "raw": _pretty_bytes(resp)}

    data = resp[data_start:data_end]
    checksum = resp[data_end]
    endb = resp[data_end + 1]

    return {
        "ok": bool(endb == END_BYTE),
        "device": dev,
        "class": class_addr,
        "subclass": subclass_addr,
        "rw": rw,
        "data": list(data),
        "checksum": checksum,
        "end": endb,
        "raw": _pretty_bytes(resp),
    }


class SerialLink:
    """
    Persistent, thread-safe serial link. USB-serial adapters (esp. PL2303)
    are often more reliable when the port stays open vs reopen-per-command.
    """
    def __init__(self):
        self._lock = threading.Lock()
        self._ser = None
        self._port = None

    def close(self):
        with self._lock:
            if self._ser is not None:
                try:
                    self._ser.close()
                except Exception:
                    pass
            self._ser = None
            self._port = None

    def _ensure_open(self, port: str, baud: int, timeout: float):
        try:
            import serial  # type: ignore
        except Exception as exc:
            raise RuntimeError("pyserial is required") from exc

        # If port changed, reopen
        if self._ser is not None and self._port != port:
            logger.info(f"[SERIAL] Port changed from {self._port} to {port}, closing old connection")
            self.close()

        if self._ser is None:
            logger.info(f"[SERIAL] Opening port {port} at {baud} baud with {timeout}s timeout")
            if not os.path.exists(port):
                logger.error(f"[SERIAL] Port does not exist: {port}")
                raise RuntimeError(f"Serial port not found: {port}")

            try:
                self._ser = serial.Serial(
                    port=port,
                    baudrate=baud,
                    timeout=timeout,
                    write_timeout=timeout,
                    rtscts=False,
                    dsrdtr=False,
                )
                logger.info(f"[SERIAL] Successfully opened {port}")
            except Exception as e:
                logger.error(f"[SERIAL] Failed to open {port}: {e}")
                raise
            
            self._port = port

            # Put lines in a known state (helps some adapters/modules)
            try:
                self._ser.dtr = True
                self._ser.rts = False
                logger.debug(f"[SERIAL] Set DTR/RTS state")
            except Exception as e:
                logger.warning(f"[SERIAL] Could not set DTR/RTS: {e}")
                pass

    def roundtrip(self, pkt: bytes, port: str, baud: int, timeout: float) -> Tuple[bytes, bytes]:
        """
        Write pkt, then read until END_BYTE (0xFF) or timeout.
        Returns (sent_bytes, raw_response_bytes).
        """
        deadline = time.monotonic() + max(0.05, float(timeout))

        with self._lock:
            try:
                self._ensure_open(port, baud, timeout)
            except Exception as e:
                logger.error(f"[SERIAL] Failed to open port {port}: {e}")
                raise
            
            ser = self._ser
            assert ser is not None

            # Flush stale input/output so we read the reply to THIS command
            try:
                ser.reset_input_buffer()
                ser.reset_output_buffer()
            except Exception as e:
                logger.warning(f"[SERIAL] Could not flush buffers: {e}")
                pass

            logger.debug(f"[SERIAL] Writing {len(pkt)} bytes: {_pretty_bytes(pkt)}")
            ser.write(pkt)
            ser.flush()

            # brief settle time for PL2303 + slower thermal modules
            time.sleep(0.05)

            buf = bytearray()
            read_start = time.monotonic()
            while time.monotonic() < deadline and len(buf) < SERIAL_READ_BYTES:
                b = ser.read(1)
                if not b:
                    continue
                buf.extend(b)
                logger.debug(f"[SERIAL] Received byte: {b[0]:02X}")
                if b[0] == END_BYTE:
                    logger.debug(f"[SERIAL] Got END_BYTE, stopping read")
                    break
            
            elapsed = time.monotonic() - read_start
            logger.info(f"[SERIAL] Read {len(buf)} bytes in {elapsed:.3f}s: {_pretty_bytes(bytes(buf)) if buf else '(empty)'}")
            return pkt, bytes(buf)


_serial_link = SerialLink()


def _serial_roundtrip(pkt: bytes, port_override: Optional[str] = None) -> Tuple[bytes, bytes]:
    port = str(port_override) if port_override else SERIAL_PORT
    logger.info(f"[SERIAL] Sending packet to port {port}: {_pretty_bytes(pkt)}")
    try:
        sent, resp = _serial_link.roundtrip(pkt, port=port, baud=SERIAL_BAUD, timeout=SERIAL_TIMEOUT)
        logger.info(f"[SERIAL] Response received: {_pretty_bytes(resp)} (len={len(resp)})")
        if not resp:
            logger.warning(f"[SERIAL] Empty response from port {port}!")
        return sent, resp
    except Exception as exc:
        logger.error(f"[SERIAL] Initial attempt failed: {exc}")
        # Retry once after forcing a reopen (handles transient USB hiccups)
        try:
            logger.info(f"[SERIAL] Retrying after port reset...")
            _serial_link.close()
            sent, resp = _serial_link.roundtrip(pkt, port=port, baud=SERIAL_BAUD, timeout=SERIAL_TIMEOUT)
            logger.info(f"[SERIAL] Retry successful: {_pretty_bytes(resp)}")
            return sent, resp
        except Exception as retry_exc:
            logger.error(f"[SERIAL] Retry failed: {retry_exc}")
            raise RuntimeError(f"Serial write failed: {exc}") from exc


def read_palette(port_override: Optional[str] = None) -> Dict[str, object]:
    pkt = _build_packet(CLASS_ADDR_PALETTE, SUBCLASS_ADDR_PALETTE, RW_READ, b"")
    sent, resp = _serial_roundtrip(pkt, port_override)
    parsed = _parse_reply(resp)
    palette_code = None
    palette_name = None
    if parsed.get("ok") and parsed.get("data"):
        palette_code = int(parsed["data"][0])
        inv = {v: k for k, v in PALETTES.items()}
        palette_name = inv.get(palette_code)
    return {
        "sent": _pretty_bytes(sent),
        "response": _pretty_bytes(resp),
        "parsed": parsed,
        "palette_code": palette_code,
        "palette_name": palette_name,
    }


def set_u8_setting(class_addr: int, subclass_addr: int, value: int, port_override: Optional[str] = None) -> Dict[str, object]:
    value = int(max(0, min(255, value)))
    pkt = _build_packet(class_addr, subclass_addr, RW_WRITE, bytes([value]))
    sent, resp = _serial_roundtrip(pkt, port_override)
    return {"sent": _pretty_bytes(sent), "response": _pretty_bytes(resp), "parsed": _parse_reply(resp), "value": value}


def read_u8_setting(class_addr: int, subclass_addr: int, port_override: Optional[str] = None) -> Dict[str, object]:
    pkt = _build_packet(class_addr, subclass_addr, RW_READ, b"")
    sent, resp = _serial_roundtrip(pkt, port_override)
    parsed = _parse_reply(resp)
    value = None
    if parsed.get("ok") and parsed.get("data"):
        value = int(parsed["data"][0])
    return {"sent": _pretty_bytes(sent), "response": _pretty_bytes(resp), "parsed": parsed, "value": value}


def set_detail_enhance(value: int, port_override: Optional[str] = None) -> Dict[str, object]:
    value = int(max(0, min(100, value)))
    return set_u8_setting(CLASS_ADDR_IMAGE, SUBCLASS_ADDR_DETAIL_ENHANCE, value, port_override)


def set_flip(mode: int, port_override: Optional[str] = None) -> Dict[str, object]:
    # 0: none, 1: center mirror, 2: left/right, 3: up/down
    mode = int(max(0, min(3, mode)))
    return set_u8_setting(CLASS_ADDR_IMAGE, SUBCLASS_ADDR_MIRROR, mode, port_override)

def set_auto_shutter(mode: int, port_override: Optional[str] = None) -> Dict[str, object]:
    # 0-3: different auto shutter modes
    mode = int(max(0, min(3, mode)))
    return set_u8_setting(CLASS_ADDR_AUTO_SHUTTER, SUBCLASS_ADDR_AUTO_SHUTTER, mode, port_override)

def set_static_denoise(value: int, port_override: Optional[str] = None) -> Dict[str, object]:
    # 0-100 range
    value = int(max(0, min(100, value)))
    return set_u8_setting(CLASS_ADDR_DENOISE, SUBCLASS_ADDR_STATIC_DENOISE, value, port_override)

def set_dynamic_denoise(value: int, port_override: Optional[str] = None) -> Dict[str, object]:
    # 0-100 range
    value = int(max(0, min(100, value)))
    return set_u8_setting(CLASS_ADDR_DENOISE, SUBCLASS_ADDR_DYNAMIC_DENOISE, value, port_override)

def factory_reset(port_override: Optional[str] = None) -> Dict[str, object]:
    # Write-only command, no data
    return set_u8_setting(CLASS_ADDR_SETTINGS, SUBCLASS_ADDR_FACTORY_RESET, 0, port_override)

def save_settings(port_override: Optional[str] = None) -> Dict[str, object]:
    # Write-only command, no data
    return set_u8_setting(CLASS_ADDR_SETTINGS, SUBCLASS_ADDR_SAVE_SETTINGS, 0, port_override)

def _pretty_bytes(buf: bytes) -> str:
    return " ".join(f"{b:02X}" for b in buf)


def apply_palette(palette: str, port_override: Optional[str] = None) -> Dict[str, str]:
    if palette.lower() in PALETTES:
        palette_code = PALETTES[palette.lower()]
    else:
        try:
            palette_code = int(palette, 16) if palette.startswith("0x") else int(palette)
        except ValueError as exc:
            raise RuntimeError("Unknown palette name or code") from exc

    pkt = _palette_packet(palette_code)
    sent, resp = _serial_roundtrip(pkt, port_override)
    return {"sent": _pretty_bytes(sent), "response": _pretty_bytes(resp)}


INDEX_HTML = """
<!doctype html>
<html lang="en">
<meta charset="utf-8"/>
<title>Dual Camera Control</title>
<style>
  :root {
    --bg: #0e1117;
    --panel: #151923;
    --accent: #3ba0ff;
    --accent-2: #ff8a3b;
    --text: #e4e7ef;
    --muted: #9aa4bd;
    --border: #262c3a;
    --success: #6bd18e;
    --danger: #ff6b6b;
  }
  * { box-sizing: border-box; }
  body {
    margin: 0;
    padding: 24px;
    font-family: "Inter", "SF Pro Display", "Segoe UI", system-ui, -apple-system, sans-serif;
    background: radial-gradient(circle at 20% 20%, rgba(59,160,255,0.12), transparent 25%),
                radial-gradient(circle at 80% 0%, rgba(255,138,59,0.10), transparent 25%),
                var(--bg);
    color: var(--text);
    min-height: 100vh;
  }
  h1 { margin: 0 0 8px; font-size: 28px; letter-spacing: -0.02em; }
  p.subtitle { margin: 0 0 24px; color: var(--muted); }
  .grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(320px, 1fr));
    gap: 16px;
    align-items: start;
  }
  .card {
    background: linear-gradient(145deg, rgba(255,255,255,0.02), rgba(255,255,255,0.01));
    border: 1px solid var(--border);
    border-radius: 14px;
    padding: 16px;
    box-shadow: 0 14px 40px rgba(0,0,0,0.35);
  }
  .card h2 { margin: 0; font-size: 20px; }
  .card header {
    display: flex;
    align-items: center;
    justify-content: space-between;
    margin-bottom: 12px;
  }
  .status {
    font-size: 12px;
    padding: 4px 8px;
    border-radius: 999px;
    border: 1px solid var(--border);
    color: var(--muted);
  }
  .status.running { color: var(--success); border-color: rgba(107,209,142,0.3); }
  .status.stopped { color: var(--danger); border-color: rgba(255,107,107,0.3); }
  .preview {
    background: #0b0d12;
    border: 1px solid var(--border);
    border-radius: 10px;
    overflow: hidden;
    display: flex;
    align-items: center;
    justify-content: center;
    margin-bottom: 12px;
    max-width: 100%;
    width: 100%;
  }
  .preview-meta { margin-top: 6px; font-size: 12px; color: var(--muted); text-align: center; }
  .preview img { max-width: 100%; height: auto; width: 100%; object-fit: contain; background: #080a10; }
  .controls { display: flex; gap: 8px; flex-wrap: wrap; }
  button {
    background: var(--panel);
    color: var(--text);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 8px 12px;
    font-weight: 600;
    cursor: pointer;
    transition: transform 120ms ease, box-shadow 120ms ease, border-color 120ms ease;
  }
  button:hover { transform: translateY(-1px); border-color: rgba(59,160,255,0.4); box-shadow: 0 6px 20px rgba(0,0,0,0.25); }
  button:active { transform: translateY(0); }
  button.primary { background: linear-gradient(135deg, #3ba0ff, #4bc9ff); color: #08111c; border: none; }
  button.destructive { border-color: rgba(255,107,107,0.4); color: var(--danger); }
  .palette-row { display: flex; gap: 8px; margin-top: 10px; flex-wrap: wrap; }
  select, input[type="text"] {
    background: var(--panel);
    color: var(--text);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 8px 10px;
    min-width: 140px;
  }
  .log { margin-top: 8px; font-size: 12px; color: var(--muted); white-space: pre-line; }
  .chip {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    padding: 6px 10px;
    border-radius: 999px;
    background: rgba(255,255,255,0.04);
    border: 1px solid var(--border);
    font-size: 12px;
    color: var(--muted);
  }
  .sensor-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(120px, 1fr));
    gap: 10px;
    margin-top: 10px;
  }
    .sensor-pill {
        padding: 10px 12px;
        border-radius: 10px;
        border: 1px solid var(--border);
        background: rgba(255,255,255,0.03);
    display: flex;
    flex-direction: column;
    gap: 4px;
  }
  .sensor-pill .label { font-size: 12px; color: var(--muted); }
  .sensor-pill .value { font-size: 18px; font-weight: 700; }

  /* Modal styles */
  .modal-backdrop {
    display: none;
    position: fixed;
    top: 0;
    left: 0;
    width: 100%;
    height: 100%;
    background: rgba(0, 0, 0, 0.85);
    backdrop-filter: blur(4px);
    z-index: 1000;
    animation: fadeIn 150ms ease-out;
  }
  .modal-backdrop.active {
    display: flex;
    align-items: center;
    justify-content: center;
  }
  @keyframes fadeIn {
    from { opacity: 0; }
    to { opacity: 1; }
  }
  .modal-container {
    position: relative;
    max-width: 90%;
    max-height: 90%;
    background: var(--panel);
    border: 1px solid var(--border);
    border-radius: 16px;
    padding: 0;
    overflow: hidden;
    box-shadow: 0 20px 60px rgba(0, 0, 0, 0.6);
    animation: slideUp 200ms ease-out;
  }
  @keyframes slideUp {
    from { transform: translateY(20px); opacity: 0; }
    to { transform: translateY(0); opacity: 1; }
  }
  .modal-content {
    width: 100%;
    height: 100%;
    display: flex;
    align-items: center;
    justify-content: center;
    background: #080a10;
  }
  .modal-content img {
    max-width: 100%;
    max-height: 100%;
    width: auto;
    height: auto;
    object-fit: contain;
  }
  .modal-close-hint {
    position: absolute;
    top: 12px;
    right: 12px;
    font-size: 12px;
    color: var(--muted);
    opacity: 0.6;
    pointer-events: none;
  }
  .card[draggable="true"] {
    cursor: grab;
    transition: opacity 150ms ease;
  }
  .card[draggable="true"]:hover {
    opacity: 0.85;
  }
  .card.dragging {
    opacity: 0.4;
    cursor: grabbing;
  }
  .card.drag-over {
    border-color: var(--accent);
    box-shadow: 0 0 0 2px rgba(255, 138, 59, 0.2);
  }

</style>
<body>
  <header style="display:flex;align-items:center;justify-content:space-between;gap:12px;flex-wrap:wrap;">
    <div>
      <h1 style="margin:0;">Project Sauron</h1>
      <p class="subtitle" style="margin:4px 0 0;"></p>
    </div>
    <div style="display:flex;gap:8px;align-items:center;">
      <label style="font-size:12px;color:var(--muted);white-space:nowrap;">FPS:</label>
      <input type="range" id="fps-slider" min="1" max="60" value="10" style="width:100px;cursor:pointer;">
      <span id="fps-display" style="font-size:12px;color:var(--text);min-width:30px;text-align:right;">10</span>
    </div>
    <button id="swap-feeds" style="background:linear-gradient(135deg, #ff8a3b, #ffb347); color:#0b0d12; border:none; padding:10px 14px; border-radius:10px; font-weight:700;">Swap feeds</button>
  </header>
  <div class="grid">
    <section class="card" id="card-rgb" draggable="true" data-camera="rgb">
      <header>
        <h2>RGB Camera</h2>
        <span class="status stopped" id="status-rgb">stopped</span>
      </header>
      <div class="preview">
        <img id="preview-rgb" alt="RGB preview"/>
      </div>
      <div class="controls">
        <button class="primary" data-camera="rgb" data-action="start">Start</button>
        <button data-camera="rgb" data-action="stop" class="destructive">Stop</button>
        <button data-camera="rgb" data-action="reconnect">Reconnect</button>
      </div>
    </section>

    <section class="card" id="card-thermal" draggable="true" data-camera="thermal">
      <header>
        <h2>Thermal Camera</h2>
        <span class="status stopped" id="status-thermal">stopped</span>
      </header>
      <div class="preview">
        <img id="preview-thermal" alt="Thermal preview"/>
      </div>
      <div class="controls">
        <button class="primary" data-camera="thermal" data-action="start">Start</button>
        <button data-camera="thermal" data-action="stop" class="destructive">Stop</button>
        <button data-camera="thermal" data-action="reconnect">Reconnect</button>
      </div>

      <div style="margin-top: 16px; padding-top: 16px; border-top: 1px solid var(--border);">
        <h3 style="margin: 0 0 12px; font-size: 13px; font-weight: 700; color: var(--text); text-transform: uppercase; letter-spacing: 0.05em;">Imaging</h3>
        
        <div style="display: flex; gap: 8px; margin-bottom: 12px; flex-wrap: wrap;">
          <select id="palette-select" aria-label="Palette" style="flex: 1; min-width: 120px;">
            <option disabled selected>Loading palettes...</option>
          </select>
          <select id="serial-port-select" aria-label="Serial port" style="flex: 0.6; min-width: 80px;">
            <option value="">Auto</option>
          </select>
          <button id="apply-palette" style="flex: 0.4; min-width: 60px;">Apply</button>
        </div>

        <div style="display: flex; gap: 6px; margin-bottom: 12px; flex-wrap: wrap;">
          <button id="detail-high" style="flex: 1; min-width: 50px;">Detail+</button>
          <button id="detail-low" style="flex: 1; min-width: 50px;">Detail−</button>
          <button id="read-palette" style="flex: 1; min-width: 50px;">Read</button>
        </div>

        <div style="display: flex; gap: 6px; margin-bottom: 12px; flex-wrap: wrap;">
          <button data-flip="0" style="flex: 1; min-width: 50px;">Flip off</button>
          <button data-flip="2" style="flex: 1; min-width: 50px;">Flip ↔</button>
          <button data-flip="3" style="flex: 1; min-width: 50px;">Flip ↕</button>
        </div>

        <div style="margin-top: 12px;">
          <div style="margin-bottom: 12px;">
            <label style="display: block; font-size: 12px; color: var(--muted); margin-bottom: 6px;">Brightness</label>
            <div style="display: flex; gap: 8px; align-items: center;">
              <input type="range" id="brightness-slider" min="0" max="255" value="128" style="flex: 1; cursor: pointer;"/>
              <span id="brightness-value" style="font-size: 12px; color: var(--text); min-width: 28px; text-align: right;">128</span>
            </div>
          </div>
          <div>
            <label style="display: block; font-size: 12px; color: var(--muted); margin-bottom: 6px;">Contrast</label>
            <div style="display: flex; gap: 8px; align-items: center;">
              <input type="range" id="contrast-slider" min="0" max="255" value="128" style="flex: 1; cursor: pointer;"/>
              <span id="contrast-value" style="font-size: 12px; color: var(--text); min-width: 28px; text-align: right;">128</span>
            </div>
          </div>
          <div style="margin-top: 12px;">
            <label style="display: block; font-size: 12px; color: var(--muted); margin-bottom: 6px;">Static Denoise</label>
            <div style="display: flex; gap: 8px; align-items: center;">
              <input type="range" id="static-denoise-slider" min="0" max="100" value="50" style="flex: 1; cursor: pointer;"/>
              <span id="static-denoise-value" style="font-size: 12px; color: var(--text); min-width: 28px; text-align: right;">50</span>
            </div>
          </div>
          <div style="margin-top: 12px;">
            <label style="display: block; font-size: 12px; color: var(--muted); margin-bottom: 6px;">Dynamic Denoise</label>
            <div style="display: flex; gap: 8px; align-items: center;">
              <input type="range" id="dynamic-denoise-slider" min="0" max="100" value="50" style="flex: 1; cursor: pointer;"/>
              <span id="dynamic-denoise-value" style="font-size: 12px; color: var(--text); min-width: 28px; text-align: right;">50</span>
            </div>
          </div>
          <div style="margin-top: 12px;">
            <label style="display: block; font-size: 12px; color: var(--muted); margin-bottom: 6px;">Auto Shutter Mode</label>
            <div style="display: flex; gap: 4px;">
              <button data-shutter="0" style="flex: 1; min-width: 40px;">Mode 0</button>
              <button data-shutter="1" style="flex: 1; min-width: 40px;">Mode 1</button>
              <button data-shutter="2" style="flex: 1; min-width: 40px;">Mode 2</button>
              <button data-shutter="3" style="flex: 1; min-width: 40px;">Mode 3</button>
            </div>
          </div>
          <div style="display: flex; gap: 8px; margin-top: 12px;">
            <button id="save-settings" style="flex: 1; background: rgba(34, 197, 94, 0.2); color: #86efac; border-color: rgba(34, 197, 94, 0.3);">Save Settings</button>
            <button id="factory-reset" style="flex: 1; background: rgba(239, 68, 68, 0.2); color: #fca5a5; border-color: rgba(239, 68, 68, 0.3);">Factory Reset</button>
          </div>
        </div>
      </div>

      <div class="log" id="palette-log" style="margin-top: 12px; font-size: 11px; line-height: 1.4;"></div>
    </section>

    <section class="card" id="card-weather" draggable="true">
      <header>
        <h2>Weather (SCD41)</h2>
        <span class="chip">CO₂ / Temp / Humidity</span>
      </header>
      <p class="subtitle">Reserved for Adafruit SCD41 sensor readings when connected.</p>
      <div class="sensor-grid" id="sensor-readings">
        <div class="sensor-pill">
          <span class="label">CO₂</span>
          <span class="value" id="sensor-co2">—</span>
        </div>
        <div class="sensor-pill">
          <span class="label">Temperature</span>
          <span class="value" id="sensor-temp">—</span>
        </div>
        <div class="sensor-pill">
          <span class="label">Humidity</span>
          <span class="value" id="sensor-humidity">—</span>
        </div>
      </div>
      <div class="log" id="sensor-log">Connect the SCD41 and expose an API (e.g., /api/sensor/scd41) to populate these values.</div>
    </section>
  </div>

  <script>
    // Drag and drop reordering
    let draggedCard = null;
    const cards = document.querySelectorAll(".card[draggable='true']");
    const grid = document.querySelector(".grid");

    cards.forEach(card => {
      card.addEventListener("dragstart", (e) => {
        draggedCard = card;
        card.classList.add("dragging");
        e.dataTransfer.effectAllowed = "move";
      });

      card.addEventListener("dragend", (e) => {
        card.classList.remove("dragging");
        cards.forEach(c => c.classList.remove("drag-over"));
        draggedCard = null;
      });

      card.addEventListener("dragover", (e) => {
        e.preventDefault();
        e.dataTransfer.dropEffect = "move";
        if (card !== draggedCard) {
          card.classList.add("drag-over");
        }
      });

      card.addEventListener("dragleave", (e) => {
        card.classList.remove("drag-over");
      });

      card.addEventListener("drop", (e) => {
        e.preventDefault();
        if (card !== draggedCard) {
          // Determine which card is before the other
          const allCards = [...grid.querySelectorAll(".card[draggable='true']")];
          const draggedIndex = allCards.indexOf(draggedCard);
          const targetIndex = allCards.indexOf(card);
          
          if (draggedIndex < targetIndex) {
            card.parentNode.insertBefore(draggedCard, card.nextSibling);
          } else {
            card.parentNode.insertBefore(draggedCard, card);
          }
          saveCardOrder();
        }
        card.classList.remove("drag-over");
      });
    });

    function saveCardOrder() {
      const order = [...grid.querySelectorAll(".card[draggable='true']")].map(c => c.id);
      localStorage.setItem("cardOrder", JSON.stringify(order));
    }

    function restoreCardOrder() {
      const saved = localStorage.getItem("cardOrder");
      if (saved) {
        try {
          const order = JSON.parse(saved);
          const allCards = [...grid.querySelectorAll(".card[draggable='true']")];
          const cardMap = new Map(allCards.map(c => [c.id, c]));
          
          order.forEach(id => {
            const card = cardMap.get(id);
            if (card) grid.appendChild(card);
          });
        } catch (e) {
          console.error("Failed to restore card order:", e);
        }
      }
    }

    // Restore order on page load
    restoreCardOrder();

    const baseStreamEndpoints = { rgb: "/stream", thermal: "/thermal" };
    let streamEndpoints = { ...baseStreamEndpoints };
    const paletteOptions = [
      "white_hot","black_hot","fusion_1","rainbow","fusion_2","iron_red_1","iron_red_2",
      "dark_brown","color_1","color_2","ice_fire","rain","green_hot","red_hot","deep_blue"
    ];

    function setPreview(camera, running) {
      const img = document.getElementById(`preview-${camera}`);
      const status = document.getElementById(`status-${camera}`);
      if (!img || !status) return;

      if (running) {
        const nextUrl = `${streamEndpoints[camera]}?t=${Date.now()}`;

        // Force-close any existing MJPEG connection
        img.removeAttribute("src");
        img.src = "about:blank";

        // Give the browser time to abort the old stream
        setTimeout(() => {
          img.src = nextUrl;
        }, 150);

        status.textContent = "running";
        status.classList.add("running");
        status.classList.remove("stopped");
      } else {
        img.removeAttribute("src");
        img.src = "about:blank";
        status.textContent = "stopped";
        status.classList.add("stopped");
        status.classList.remove("running");
      }
    }

    async function cameraAction(camera, action) {
      try {
        const res = await fetch(`/api/camera/${camera}/${action}`, { method: "POST" });
        const json = await res.json();
        if (!res.ok) throw new Error(json.error || "Request failed");
        setPreview(camera, json.running);
      } catch (err) {
        console.error(err);
        alert(err.message || "Camera action failed");
      }
    }

    function populatePalettes(options) {
      const select = document.getElementById("palette-select");
      if (!select) return;
      const seen = new Set();
      const merged = [];
      (options || []).forEach(name => {
        if (name && !seen.has(name)) {
          seen.add(name);
          merged.push(name);
        }
      });
      if (!merged.length) return;
      select.innerHTML = "";
      merged.forEach(name => {
        const opt = document.createElement("option");
        opt.value = name;
        opt.textContent = name.replace(/_/g, " ");
        select.appendChild(opt);
      });
    }

    async function syncStatus() {
      try {
        const [statusRes, infoRes] = await Promise.all([fetch("/api/status"), fetch("/api/camera/info")]);
        const json = await statusRes.json();
        const info = infoRes.ok ? await infoRes.json() : {};

        ["rgb", "thermal"].forEach(name => {
          setPreview(name, json.cameras?.[name]?.running);
          const shape = info?.[name]?.shape;
          
          // Apply dynamic aspect-ratio based on actual frame dimensions
          const previewEl = document.getElementById(`preview-${name}`);
          if (previewEl && previewEl.parentElement && shape && shape.width && shape.height) {
            const aspectRatio = shape.width / shape.height;
            previewEl.parentElement.style.aspectRatio = aspectRatio;
          }
        });

        if (Array.isArray(json.palette_options)) {
          populatePalettes(json.palette_options);
        }
        if (json.available_ports && Array.isArray(json.available_ports)) {
          populateSerialPorts(json.available_ports, json.default_port);
        }
      } catch (err) {
        console.warn("Status sync failed", err);
      }
    }

    function populateSerialPorts(ports, defaultPort) {
      const select = document.getElementById("serial-port-select");
      if (!select) return;
      const savedPort = localStorage.getItem("selected-serial-port") || "";
      select.innerHTML = '';
      const autoOpt = document.createElement("option");
      autoOpt.value = "";
      autoOpt.textContent = "Auto";
      select.appendChild(autoOpt);
      ports.forEach(port => {
        const opt = document.createElement("option");
        opt.value = port;
        opt.textContent = port;
        select.appendChild(opt);
      });
      if (savedPort && ports.includes(savedPort)) {
        select.value = savedPort;
      } else {
        select.value = "";
      }
    }

    function getSelectedSerialPort() {
      const select = document.getElementById("serial-port-select");
      const port = select ? select.value : "";
      if (port) localStorage.setItem("selected-serial-port", port);
      return port;
    }

    document.addEventListener("DOMContentLoaded", () => {
      const portSelect = document.getElementById("serial-port-select");
      if (portSelect) {
        portSelect.addEventListener("change", () => {
          const port = portSelect.value;
          if (port) {
            localStorage.setItem("selected-serial-port", port);
          } else {
            localStorage.removeItem("selected-serial-port");
          }
        });
      }
    });

    document.querySelectorAll("button[data-camera]").forEach(btn => {
      btn.addEventListener("click", () => {
        const camera = btn.dataset.camera;
        const action = btn.dataset.action;
        cameraAction(camera, action);
      });
    });

    document.getElementById("apply-palette").addEventListener("click", async () => {
      const palette = document.getElementById("palette-select").value;
      const port = getSelectedSerialPort();
      const body = { palette };
      if (port) body.port = port;
      const logEl = document.getElementById("palette-log");
      logEl.textContent = "Sending palette command...";
      logEl.style.color = "var(--muted)";
      try {
        const res = await fetch("/api/palette", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(body)
        });
        const json = await res.json();
        if (!res.ok) throw new Error(json.error || "Palette request failed");
        logEl.textContent = `Palette set. Sent: ${json.sent || ""}\\nResponse: ${json.response || "(none)"}`;
      } catch (err) {
        logEl.textContent = err.message || "Palette request failed";
      }
    });

    // Palette readback
    document.getElementById("read-palette").addEventListener("click", async () => {
      const port = getSelectedSerialPort();
      const logEl = document.getElementById("palette-log");
      logEl.textContent = "Reading current palette...";
      try {
        const qs = port ? `?port=${encodeURIComponent(port)}` : "";
        const res = await fetch(`/api/thermal/palette${qs}`);
        const json = await res.json();
        if (!res.ok) throw new Error(json.error || "Read palette failed");

        if (json.palette_name) {
          const sel = document.getElementById("palette-select");
          sel.value = json.palette_name;
        }

        logEl.textContent = `Palette read. Name: ${json.palette_name || "(unknown)"} Code: ${json.palette_code ?? "?"}\\nSent: ${json.sent || ""}\\nResponse: ${json.response || "(none)"}`;
      } catch (err) {
        logEl.textContent = err.message || "Read palette failed";
      }
    });

    // Detail presets
    async function setDetailPreset(kind) {
      const port = getSelectedSerialPort();
      const logEl = document.getElementById("palette-log");
      logEl.textContent = `Setting ${kind} detail...`;
      try {
        const body = {};
        if (port) body.port = port;
        const res = await fetch(`/api/thermal/detail/${kind}`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(body)
        });
        const json = await res.json();
        if (!res.ok) throw new Error(json.error || "Detail request failed");
        logEl.textContent = `${kind} detail applied. Value: ${json.value}.\\nSent: ${json.sent || ""}\\nResponse: ${json.response || "(none)"}`;
      } catch (err) {
        logEl.textContent = err.message || "Detail request failed";
      }
    }
    document.getElementById("detail-high").addEventListener("click", () => setDetailPreset("high"));
    document.getElementById("detail-low").addEventListener("click", () => setDetailPreset("low"));

    // Flip buttons
    document.querySelectorAll("button[data-flip]").forEach(btn => {
      btn.addEventListener("click", async () => {
        const mode = Number(btn.dataset.flip);
        const port = getSelectedSerialPort();
        const logEl = document.getElementById("palette-log");
        logEl.textContent = "Setting flip mode...";
        try {
          const body = { mode };
          if (port) body.port = port;
          const res = await fetch("/api/thermal/flip", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(body)
          });
          const json = await res.json();
          if (!res.ok) throw new Error(json.error || "Flip request failed");
          logEl.textContent = `Flip mode set to ${json.mode}.\\nSent: ${json.sent || ""}\\nResponse: ${json.response || "(none)"}`;
        } catch (err) {
          logEl.textContent = err.message || "Flip request failed";
        }
      });
    });

    // Add a guard to prevent overlapping swaps.
    let swapInProgress = false;

    document.getElementById("swap-feeds").addEventListener("click", () => {
      if (swapInProgress) return;
      swapInProgress = true;

      const rgbRunning = document.getElementById("status-rgb")?.classList.contains("running");
      const thermalRunning = document.getElementById("status-thermal")?.classList.contains("running");

      // Pause current previews while swap happens server-side
      setPreview("rgb", false);
      setPreview("thermal", false);

      fetch("/api/camera/swap", { method: "POST" })
        .then(async res => {
          const json = await res.json();
          if (!res.ok) throw new Error(json.error || "Swap failed");
          // After swap, reset stream endpoints to defaults
          streamEndpoints = { ...baseStreamEndpoints };
          if (rgbRunning) setPreview("rgb", true);
          if (thermalRunning) setPreview("thermal", true);
        })
        .catch(err => {
          alert(err.message || "Swap failed");
        })
        .finally(() => {
          swapInProgress = false;
          // Hide the swap button after successful swap
          document.getElementById("swap-feeds").style.display = "none";
          // refresh statuses so UI matches server state
          syncStatus();
        });
    });

    // FPS Control Slider
    const fpsSlider = document.getElementById("fps-slider");
    const fpsDisplay = document.getElementById("fps-display");
    
    // Initialize FPS display on page load
    fetch("/api/fps")
      .then(res => res.json())
      .then(data => {
        fpsSlider.value = data.fps;
        fpsDisplay.textContent = Math.round(data.fps);
      })
      .catch(err => console.error("Failed to get FPS:", err));
    
    fpsSlider.addEventListener("input", async (e) => {
      const fps = parseInt(e.target.value);
      fpsDisplay.textContent = fps;
      try {
        const res = await fetch("/api/fps", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ fps: fps })
        });
        if (!res.ok) {
          const json = await res.json();
          throw new Error(json.error || "Failed to set FPS");
        }
        const data = await res.json();
        fpsDisplay.textContent = Math.round(data.fps);
      } catch (err) {
        console.error("FPS error:", err);
      }
    });

    // Placeholder for SCD41; fill from a future endpoint if desired
    async function syncSensor() {
      try {
        const res = await fetch("/api/sensor/scd41");
        if (!res.ok) throw new Error("not available");
        const data = await res.json();
        if (typeof data.co2 !== "undefined") document.getElementById("sensor-co2").textContent = `${data.co2} ppm`;
        if (typeof data.temp_c !== "undefined") document.getElementById("sensor-temp").textContent = `${data.temp_c} °C`;
        if (typeof data.humidity !== "undefined") document.getElementById("sensor-humidity").textContent = `${data.humidity} %`;
        document.getElementById("sensor-log").textContent = "Live SCD41 data";
      } catch (err) {
        document.getElementById("sensor-log").textContent = "Sensor endpoint not available";
      }
    }

    populatePalettes(paletteOptions);
    syncStatus();
    syncSensor();

    // Modal preview functionality
    const modalBackdrop = document.createElement("div");
    modalBackdrop.id = "modal-backdrop";
    modalBackdrop.className = "modal-backdrop";
    modalBackdrop.innerHTML = `
      <div class="modal-container">
        <div class="modal-content">
          <img id="modal-image" alt="Expanded preview"/>
        </div>
      </div>
    `;
    document.body.appendChild(modalBackdrop);

    // Open modal when clicking on preview images
    ["rgb", "thermal"].forEach(camera => {
      const img = document.getElementById(`preview-${camera}`);
      if (img) {
        img.style.cursor = "pointer";
        img.addEventListener("click", (e) => {
          e.stopPropagation();
          const modalImg = document.getElementById("modal-image");
          modalImg.src = img.src;
          modalBackdrop.classList.add("active");
        });
      }
    });

    // Close modal when clicking outside
    modalBackdrop.addEventListener("click", (e) => {
      if (e.target === modalBackdrop) {
        modalBackdrop.classList.remove("active");
      }
    });

    // Handle brightness slider
    const brightnessSlider = document.getElementById("brightness-slider");
    const brightnessValue = document.getElementById("brightness-value");
    brightnessSlider.addEventListener("input", async (e) => {
      const value = parseInt(e.target.value);
      brightnessValue.textContent = value;
      const port = getSelectedSerialPort();
      try {
        const body = { value };
        if (port) body.port = port;
        const res = await fetch("/api/thermal/brightness", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(body)
        });
        if (!res.ok) {
          const json = await res.json();
          throw new Error(json.error || "Failed to set brightness");
        }
      } catch (err) {
        console.error("Brightness error:", err);
      }
    });

    // Handle contrast slider
    const contrastSlider = document.getElementById("contrast-slider");
    const contrastValue = document.getElementById("contrast-value");
    contrastSlider.addEventListener("input", async (e) => {
      const value = parseInt(e.target.value);
      contrastValue.textContent = value;
      const port = getSelectedSerialPort();
      try {
        const body = { value };
        if (port) body.port = port;
        const res = await fetch("/api/thermal/contrast", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(body)
        });
        if (!res.ok) {
          const json = await res.json();
          throw new Error(json.error || "Failed to set contrast");
        }
      } catch (err) {
        console.error("Contrast error:", err);
      }
    });

    // Static Denoise slider
    const staticDenoiseSlider = document.getElementById("static-denoise-slider");
    const staticDenoiseValue = document.getElementById("static-denoise-value");
    staticDenoiseSlider.addEventListener("input", async (e) => {
      const value = parseInt(e.target.value);
      staticDenoiseValue.textContent = value;
      const port = getSelectedSerialPort();
      try {
        const body = { value };
        if (port) body.port = port;
        const res = await fetch("/api/thermal/static-denoise", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(body)
        });
        if (!res.ok) {
          const json = await res.json();
          throw new Error(json.error || "Failed to set static denoise");
        }
      } catch (err) {
        console.error("Static denoise error:", err);
      }
    });

    // Dynamic Denoise slider
    const dynamicDenoiseSlider = document.getElementById("dynamic-denoise-slider");
    const dynamicDenoiseValue = document.getElementById("dynamic-denoise-value");
    dynamicDenoiseSlider.addEventListener("input", async (e) => {
      const value = parseInt(e.target.value);
      dynamicDenoiseValue.textContent = value;
      const port = getSelectedSerialPort();
      try {
        const body = { value };
        if (port) body.port = port;
        const res = await fetch("/api/thermal/dynamic-denoise", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(body)
        });
        if (!res.ok) {
          const json = await res.json();
          throw new Error(json.error || "Failed to set dynamic denoise");
        }
      } catch (err) {
        console.error("Dynamic denoise error:", err);
      }
    });

    // Auto Shutter Mode buttons
    document.querySelectorAll("[data-shutter]").forEach(btn => {
      btn.addEventListener("click", async (e) => {
        const mode = parseInt(e.target.dataset.shutter);
        const port = getSelectedSerialPort();
        try {
          const body = { value: mode };
          if (port) body.port = port;
          const res = await fetch("/api/thermal/auto-shutter", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(body)
          });
          if (!res.ok) {
            const json = await res.json();
            throw new Error(json.error || "Failed to set auto shutter");
          }
        } catch (err) {
          console.error("Auto shutter error:", err);
        }
      });
    });

    // Save Settings button
    document.getElementById("save-settings").addEventListener("click", async () => {
      const port = getSelectedSerialPort();
      try {
        const body = {};
        if (port) body.port = port;
        const res = await fetch("/api/thermal/save-settings", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(body)
        });
        if (!res.ok) {
          const json = await res.json();
          throw new Error(json.error || "Failed to save settings");
        }
        const logEl = document.getElementById("palette-log");
        logEl.textContent = "✓ Settings saved successfully";
      } catch (err) {
        console.error("Save settings error:", err);
        alert(err.message || "Failed to save settings");
      }
    });

    // Factory Reset button
    document.getElementById("factory-reset").addEventListener("click", async () => {
      if (!confirm("Are you sure you want to reset to factory settings? This cannot be undone.")) {
        return;
      }
      const port = getSelectedSerialPort();
      try {
        const body = {};
        if (port) body.port = port;
        const res = await fetch("/api/thermal/factory-reset", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(body)
        });
        if (!res.ok) {
          const json = await res.json();
          throw new Error(json.error || "Failed to reset factory settings");
        }
        const logEl = document.getElementById("palette-log");
        logEl.textContent = "✓ Factory reset completed. Device may need to restart.";
      } catch (err) {
        console.error("Factory reset error:", err);
        alert(err.message || "Failed to reset factory settings");
      }
    });

  </script>
</body>
</html>
"""

@app.route('/')
def index():
    palette_options = sorted(PALETTES.keys())
    return render_template_string(
        INDEX_HTML,
        palette_options=palette_options,
        default_port=SERIAL_PORT,
    )


def _get_available_serial_ports():
    """Detect available serial ports."""
    import glob
    ports = []
    # Linux/Mac
    ports.extend(glob.glob('/dev/ttyUSB*'))
    ports.extend(glob.glob('/dev/ttyACM*'))
    ports.extend(glob.glob('/dev/cu.*'))
    ports.extend(glob.glob('/dev/tty.*'))
    # Windows
    try:
        import serial.tools.list_ports
        for port, desc, hwid in serial.tools.list_ports.comports():
            ports.append(port)
    except:
        pass
    return sorted(list(set(ports)))


@app.route("/api/status")
def api_status():
    return jsonify(
        {
            "cameras": {
                "rgb": rgb.status(),
                "thermal": thermal.status(),
            },
            "default_port": SERIAL_PORT,
            "available_ports": _get_available_serial_ports(),
            "palette_options": sorted(PALETTES.keys()),
            "current_fps": current_fps,
        }
    )


@app.route("/api/fps", methods=["GET"])
def api_get_fps():
    """Get current FPS setting"""
    return jsonify({"fps": current_fps, "min": 1, "max": 60})


@app.route("/api/fps", methods=["POST"])
def api_set_fps():
    """Set FPS on the fly"""
    global current_fps
    payload = request.get_json(force=True, silent=True) or {}
    fps = payload.get("fps")
    if fps is None:
        return jsonify({"error": "fps is required"}), 400
    try:
        fps = float(fps)
        current_fps = max(1.0, min(60.0, fps))
        return jsonify({"success": True, "fps": current_fps})
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500


@app.route("/api/camera/info")
def api_camera_info():
    """Return runtime info (including last frame shape) for both cameras."""
    try:
        return jsonify({
            "rgb": {**rgb.status(), "shape": rgb.latest_shape()},
            "thermal": {**thermal.status(), "shape": thermal.latest_shape()},
        })
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500


def _camera_action(name: str, action: str):
    try:
        cam_name, grabber = _grabber_for(name)
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 404
    try:
        if action == "start":
            grabber.start()
        elif action == "stop":
            grabber.stop()
        elif action == "reconnect":
            grabber.reconnect()
        else:
            raise ValueError("Unsupported action")
    except Exception as exc:
        return jsonify({"error": str(exc)}), 400
    return jsonify({"camera": cam_name, "running": grabber.running})


@app.route("/api/camera/<name>/start", methods=["POST"])
def api_camera_start(name):
    return _camera_action(name, "start")


@app.route("/api/camera/<name>/stop", methods=["POST"])
def api_camera_stop(name):
    return _camera_action(name, "stop")


@app.route("/api/camera/<name>/reconnect", methods=["POST"])
def api_camera_reconnect(name):
    return _camera_action(name, "reconnect")

@app.route("/api/camera/swap", methods=["POST"])
def api_camera_swap():
    try:
        result = swap_camera_devices()
    except Exception as exc:
        return jsonify({"error": str(exc)}), 400
    return jsonify(result)


@app.route("/api/palette", methods=["POST"])
def api_palette():
    payload = request.get_json(force=True, silent=True) or {}
    palette = payload.get("palette")
    port_override = payload.get("port")
    if not palette:
        return jsonify({"error": "Palette is required"}), 400
    try:
        result = apply_palette(str(palette), str(port_override) if port_override else None)
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500
    return jsonify(result)

@app.route("/api/thermal/palette", methods=["GET"])
def api_thermal_palette_read():
    port_override = request.args.get("port")
    try:
        result = read_palette(str(port_override) if port_override else None)
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500
    return jsonify(result)


# Camera crop endpoints
@app.route("/api/camera/<name>/crop", methods=["GET", "POST"])
def api_camera_crop(name):
    try:
        cam_name, grabber = _grabber_for(name)
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 404

    if request.method == "GET":
        return jsonify({"camera": cam_name, "crop": grabber.get_crop()})

    payload = request.get_json(force=True, silent=True) or {}
    top = int(payload.get("top", 0))
    bottom = int(payload.get("bottom", 0))
    left = int(payload.get("left", 0))
    right = int(payload.get("right", 0))
    try:
        grabber.set_crop(top, bottom, left, right)
    except Exception as exc:
        return jsonify({"error": str(exc)}), 400
    return jsonify({"camera": cam_name, "crop": grabber.get_crop()})


@app.route("/api/camera/<name>/crop/autovertical", methods=["POST"])
def api_camera_autovertical(name):
    payload = request.get_json(force=True, silent=True) or {}
    percent = float(payload.get("percent", 5.0))
    try:
        cam_name, grabber = _grabber_for(name)
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 404

    shape = grabber.latest_shape()
    if not shape:
        return jsonify({"error": "no frame available"}), 400

    h = int(shape.get("height", 0))
    px = int(round(h * (percent / 100.0)))
    top = bottom = px
    try:
        grabber.set_crop(top=top, bottom=bottom, left=0, right=0)
    except Exception as exc:
        return jsonify({"error": str(exc)}), 400
    return jsonify({"camera": cam_name, "crop": grabber.get_crop(), "applied_percent": percent})


@app.route("/api/thermal/detail/high", methods=["POST"])
def api_thermal_detail_high():
    payload = request.get_json(force=True, silent=True) or {}
    port_override = payload.get("port")
    try:
        result = set_detail_enhance(80, str(port_override) if port_override else None)
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500
    return jsonify(result)


@app.route("/api/thermal/detail/low", methods=["POST"])
def api_thermal_detail_low():
    payload = request.get_json(force=True, silent=True) or {}
    port_override = payload.get("port")
    try:
        result = set_detail_enhance(20, str(port_override) if port_override else None)
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500
    return jsonify(result)


@app.route("/api/thermal/flip", methods=["POST"])
def api_thermal_flip_set():
    payload = request.get_json(force=True, silent=True) or {}
    mode = payload.get("mode")
    port_override = payload.get("port")
    if mode is None:
        return jsonify({"error": "mode is required"}), 400
    try:
        result = set_flip(int(mode), str(port_override) if port_override else None)
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500
    result["mode"] = int(max(0, min(3, int(mode))))
    return jsonify(result)


@app.route("/api/thermal/brightness", methods=["POST"])
def api_thermal_brightness_set():
    payload = request.get_json(force=True, silent=True) or {}
    value = payload.get("value")
    port_override = payload.get("port")
    if value is None:
        return jsonify({"error": "value is required"}), 400
    try:
        result = set_u8_setting(CLASS_ADDR_IMAGE, SUBCLASS_ADDR_BRIGHTNESS, int(value), str(port_override) if port_override else None)
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500
    return jsonify(result)


@app.route("/api/thermal/contrast", methods=["POST"])
def api_thermal_contrast_set():
    payload = request.get_json(force=True, silent=True) or {}
    value = payload.get("value")
    port_override = payload.get("port")
    if value is None:
        return jsonify({"error": "value is required"}), 400
    try:
        result = set_u8_setting(CLASS_ADDR_IMAGE, SUBCLASS_ADDR_CONTRAST, int(value), str(port_override) if port_override else None)
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500
    return jsonify(result)


@app.route("/api/thermal/auto-shutter", methods=["POST"])
def api_thermal_auto_shutter_set():
    payload = request.get_json(force=True, silent=True) or {}
    value = payload.get("value")
    port_override = payload.get("port")
    if value is None:
        return jsonify({"error": "value is required"}), 400
    try:
        result = set_auto_shutter(int(value), str(port_override) if port_override else None)
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500
    return jsonify(result)


@app.route("/api/thermal/static-denoise", methods=["POST"])
def api_thermal_static_denoise_set():
    payload = request.get_json(force=True, silent=True) or {}
    value = payload.get("value")
    port_override = payload.get("port")
    if value is None:
        return jsonify({"error": "value is required"}), 400
    try:
        result = set_static_denoise(int(value), str(port_override) if port_override else None)
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500
    return jsonify(result)


@app.route("/api/thermal/dynamic-denoise", methods=["POST"])
def api_thermal_dynamic_denoise_set():
    payload = request.get_json(force=True, silent=True) or {}
    value = payload.get("value")
    port_override = payload.get("port")
    if value is None:
        return jsonify({"error": "value is required"}), 400
    try:
        result = set_dynamic_denoise(int(value), str(port_override) if port_override else None)
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500
    return jsonify(result)


@app.route("/api/thermal/factory-reset", methods=["POST"])
def api_thermal_factory_reset():
    port_override = request.get_json(force=True, silent=True) or {}
    try:
        result = factory_reset(str(port_override.get("port")) if port_override.get("port") else None)
        return jsonify({"success": result.get("success"), "message": "Factory reset initiated"})
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500


@app.route("/api/thermal/save-settings", methods=["POST"])
def api_thermal_save_settings():
    port_override = request.get_json(force=True, silent=True) or {}
    try:
        result = save_settings(str(port_override.get("port")) if port_override.get("port") else None)
        return jsonify({"success": result.get("success"), "message": "Settings saved"})
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500


def _start_all():
    # Start both grabbers; if thermal device is absent, keep RGB only
    try: 
        rgb.start()
    except Exception as e:
        print(f"[ERROR] RGB grabber failed to start: {e}")
    try:
        thermal.start()
    except Exception as e:
        # Log but keep server running for the RGB stream
        print(f"[WARN] Thermal device failed to open: {e}")

def _stop_all():
    try:
        _serial_link.close()
    except Exception:
        pass
    thermal.stop()
    rgb.stop()

if __name__ == '__main__':
    try:
        _start_all()
        app.run(host='0.0.0.0', port=8080, threaded=True)
    finally:
        _stop_all()

