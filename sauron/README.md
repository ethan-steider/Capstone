# Project Sauron

Real-time thermal and RGB camera control with advanced image processing and HM-TM5X thermal camera protocol support.

## Description

Project Sauron is a comprehensive thermal camera control system featuring:

- **Dual Camera Streaming**: Simultaneous RGB and thermal imaging (30 FPS default, adjustable)
- **HM-TM5X Protocol Support**: Full control of thermal camera settings via serial communication
- **Real-time Web Interface**: Modern, responsive web UI with draggable controls
- **Dual Streaming Modes**: 
  - HTTP/MJPEG (TCP) - Recommended for standard use
  - UDP - Low-latency alternative with WebSocket tunnel
- **Advanced Thermal Controls**: Brightness, contrast, denoise, auto-shutter, flip, and more

## Features

### Video Streaming
- Real-time MJPEG streaming (1-60 FPS adjustable on the fly)
- Dual camera support (RGB + Thermal)
- Frame rate control without restarting streams
- Modal preview with click-to-expand video

### Thermal Camera Control
- **Image Settings**: Brightness (0-255), Contrast (0-255), Detail (0-100)
- **Denoise**: Static (0-100), Dynamic (0-100)
- **Shutter Control**: Auto modes (0-3)
- **Flip Modes**: None, Horizontal, Vertical
- **Palette Selection**: 15+ color schemes
- **Settings Management**: Save and factory reset

### User Interface
- Draggable card layout (reorderable with drag-and-drop)
- Responsive grid layout
- Real-time slider controls with value display
- Serial port auto-detection and selection
- Dark theme optimized for low-light operation  

## Installation

### Requirements
- Python 3.7+
- OpenCV (`cv2`)
- Flask
- PySerial

### Setup

```bash
# Install dependencies
pip install opencv-python flask pyserial

# Navigate to src directory
cd /Users/ethansteider/Desktop/Capstone/Capstone/sauron/src

# Run the server
python3 sliderserver.py
```

## Usage

### HTTP Server (Recommended)

```bash
python3 sliderserver.py
# Open browser: http://localhost:8080
```

**Features:**
- Full thermal control panel
- Real-time video preview
- FPS adjustment slider in header
- Draggable cards for custom layout
- Palette selection dropdown

### UDP Server (Low-Latency Alternative)

```bash
python3 serverUDP.py
# Open clientUDP.html in browser
# Connect to ws://localhost:8081
```

**Features:**
- UDP-based video streaming (lower latency)
- WebSocket tunnel for browser compatibility
- Thermal controls via UDP commands

## API Endpoints (HTTP Server)

### Camera Control
- `POST /api/camera/rgb/start` - Start RGB stream
- `POST /api/camera/rgb/stop` - Stop RGB stream
- `POST /api/camera/thermal/start` - Start thermal stream
- `POST /api/camera/thermal/stop` - Stop thermal stream
- `POST /api/camera/swap` - Swap RGB/Thermal feeds

### Thermal Settings
- `POST /api/thermal/brightness` - Set brightness (0-255)
- `POST /api/thermal/contrast` - Set contrast (0-255)
- `POST /api/thermal/static-denoise` - Set static denoise (0-100)
- `POST /api/thermal/dynamic-denoise` - Set dynamic denoise (0-100)
- `POST /api/thermal/auto-shutter` - Set auto shutter mode (0-3)
- `POST /api/thermal/palette` - Set color palette
- `POST /api/thermal/save-settings` - Save current config
- `POST /api/thermal/factory-reset` - Factory reset

### Frame Rate Control
- `GET /api/fps` - Get current FPS
- `POST /api/fps` - Set FPS (1-60)

### Status & Info
- `GET /api/status` - Get camera and port status
- `GET /api/camera/info` - Get camera resolution info

## Configuration

Set environment variables to customize behavior:

```bash
# Camera settings
export CAM_WIDTH=1280              # RGB camera width
export CAM_HEIGHT=960             # RGB camera height
export THERMAL_WIDTH=640          # Thermal camera width
export THERMAL_HEIGHT=480         # Thermal camera height
export CAM_FPS=10                 # Default FPS (1-60)
export CAM_INDEX=0                # RGB camera device index
export THERMAL_INDEX=1            # Thermal camera device index

# Serial settings
export THERMAL_SERIAL_PORT="/dev/serial/by-id/usb-Prolific_..."
export THERMAL_SERIAL_BAUD=115200
export THERMAL_SERIAL_TIMEOUT=0.2
```

## Protocol Support

### HM-TM5X UART/CVBS Thermal Camera
Full protocol implementation with:
- Packet structure: `[0xF0][SIZE][0x36][CLASS][SUBCLASS][RW][DATA...][CHECKSUM][0xFF]`
- Dynamic checksum calculation
- Retry logic with timeout handling
- Serial communication at 115200 baud

## Testing

Run included test scripts to verify setup:

```bash
# Test single camera
python3 cameratest.py

# Test dual cameras (RGB + Thermal)
python3 dualcameratest.py

# Test thermal serial communication
python3 thermal_serial_test.py

# Test frame reading
python3 test_read.py
```

## File Structure

```
sauron/
├── README.md                    # This file
├── STRUCTURE.md                 # Detailed file organization
├── src/
│   ├── sliderserver.py         # Main HTTP server (69 KB)
│   ├── serverUDP.py            # UDP server alternative (18 KB)
│   ├── clientUDP.html          # UDP client UI (15 KB)
│   ├── camapi.py               # Camera library (1.7 KB)
│   ├── cameratest.py           # Camera test
│   ├── dualcameratest.py       # Dual camera test
│   ├── thermal_serial_test.py  # Serial communication test
│   └── test_read.py            # Frame reading test
```

## Supported Cameras

- **RGB**: Any V4L2/OpenCV-compatible USB camera
- **Thermal**: HM-TM5X series with USB serial interface

## Performance

- **Video Streaming**: ~30 FPS @ 1280x960 (RGB), 640x480 (Thermal)
- **Latency**: <100ms (HTTP), <50ms (UDP)
- **Memory**: ~50-100 MB runtime
- **Network**: <5 Mbps bandwidth (adjustable via FPS)

## Browser Compatibility

- Chrome/Chromium 60+
- Firefox 55+
- Safari 12+
- Edge 79+

## Future Roadmap

- [ ] Multi-user streaming
- [ ] Recording capability
- [ ] Advanced analytics (temperature mapping)
- [ ] Mobile app
- [ ] RTP/RTSP streaming support

## Troubleshooting

### No cameras detected
```bash
ls /dev/video*
v4l2-ctl --list-devices
```

### Serial port not found
```bash
ls /dev/serial/by-id/
# Verify thermal camera is connected via USB
```

### High CPU usage
- Reduce FPS via slider (try 10-15 FPS)
- Lower camera resolution in environment variables
- Check for background processes

### Port 8080 already in use
```bash
# Change Flask port in environment
export FLASK_PORT=8081
```

## License

Proprietary - Capstone Project

## Author

Ethan Steider - Project Sauron Capstone
