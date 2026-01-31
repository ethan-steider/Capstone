# Project Sauron Repository Structure

## Overview
Clean, organized repository for thermal camera control and video streaming.

## Current Files

### Core Servers
- **`http_thermal_server.py`** - Main HTTP/Flask server
  - MJPEG video streaming (TCP/HTTP)
  - Thermal camera control (brightness, contrast, denoise, auto-shutter, flip)
  - Palette selection and settings management
  - FPS control on the fly
  - Draggable UI cards with localStorage persistence
  - Factory reset and settings save functions

- **`udp_thermal_server.py`** - UDP-based video streaming server
  - Alternative low-latency streaming via UDP
  - WebSocket tunnel for browser compatibility
  - Frame fragmentation and reassembly
  - Same thermal control capabilities as http_thermal_server
  - Dynamic FPS adjustment

- **`udp_client.html`** - Browser client for UDP server
  - Canvas-based JPEG frame rendering
  - WebSocket connectivity
  - Thermal control sliders and buttons
  - Frame reassembly from UDP packets

### Core Libraries
- **`camera.py`** - Camera abstraction layer
  - FrameGrabber class for video capture
  - Support for multiple camera devices
  - Frame format negotiation (MJPG/raw)

### Test Scripts
- **`test_single_camera.py`** - Basic camera functionality test
- **`test_dual_cameras.py`** - Test dual camera setup (RGB + Thermal)
- **`test_thermal_serial.py`** - Thermal serial communication test
- **`test_frame_reading.py`** - Frame reading test

## Running the Servers

### HTTP Server (Recommended for local use)
```bash
python3 http_thermal_server.py
# Visit http://localhost:8080 in browser
```

### UDP Server (Low-latency streaming)
```bash
python3 udp_thermal_server.py
# Open udp_client.html in browser
# Connect to ws://localhost:8081
```

## Features

### Video Streaming
- Real-time MJPEG streaming (30 FPS default, adjustable 1-60 FPS)
- Dual camera support (RGB + Thermal)
- Dynamic FPS control without restarting

### Thermal Camera Control (HM-TM5X Protocol)
- **Image Controls**: Brightness (0-255), Contrast (0-255), Detail (0-100)
- **Denoise**: Static (0-100), Dynamic (0-100)
- **Shutter**: Auto modes (0-3)
- **Flip**: None, Horizontal, Vertical
- **Settings**: Save current config, Factory reset

### UI Features
- Draggable cards (reorderable layout)
- Responsive grid layout
- Real-time thermal control sliders
- Serial port selection with persistence
- Palette/color scheme selection
- FPS adjustment slider in header

### Protocol Support
- HM-TM5X UART/CVBS thermal camera protocol
- HTTP/TCP MJPEG streaming
- UDP with WebSocket tunnel
- Serial USB communication (115200 baud)

## Environment Variables

```bash
# Camera settings
export CAM_WIDTH=1280           # RGB camera width
export CAM_HEIGHT=960          # RGB camera height
export THERMAL_WIDTH=640       # Thermal camera width
export THERMAL_HEIGHT=480      # Thermal camera height
export CAM_FPS=10              # Default FPS
export CAM_INDEX=0             # RGB camera device index
export THERMAL_INDEX=1         # Thermal camera device index

# Serial settings
export THERMAL_SERIAL_PORT="/dev/serial/by-id/usb-Prolific_..."
export THERMAL_SERIAL_BAUD=115200
export THERMAL_SERIAL_TIMEOUT=0.2
```

## File Sizes
- `sliderserver.py` - ~69 KB (main server)
- `serverUDP.py` - ~18 KB (UDP server)
- `clientUDP.html` - ~15 KB (UDP client UI)
- `camapi.py` - ~1.7 KB (camera library)
- Test scripts - ~11 KB total

**Total: ~114 KB** (clean, minimal repository)
