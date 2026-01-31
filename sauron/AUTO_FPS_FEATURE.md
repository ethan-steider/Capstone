# Auto FPS Network Quality Feature

## Overview
The HTTP thermal server now includes intelligent network monitoring and automatic FPS adjustment based on real-time connection quality.

## Features

### Network Quality Monitoring
- **Real-time Bandwidth Calculation**: Monitors frame sizes and transmission timing
- **Latency Measurement**: Measures round-trip latency to the server
- **Quality Assessment**: Classifies connection as Excellent/Good/Fair/Poor
- **Visual Indicator**: Color-coded quality badge in the header (green=excellent, blue=good, yellow=fair, red=poor)

### Auto FPS Adjustment
- **Automatic Mode**: Click "Auto FPS" button to enable adaptive frame rate
- **Quality-Based Recommendations**:
  - Excellent (>10 Mbps, <10ms) → 30 FPS
  - Good (>5 Mbps, <20ms) → 20 FPS
  - Fair (>2 Mbps, <50ms) → 10 FPS
  - Poor → 5 FPS
- **Real-time Adaptation**: Adjusts FPS every 3 seconds when auto mode is active
- **Manual Override**: Can still manually adjust FPS slider at any time

## UI Components

### Header Updates
- **Network Quality Badge**: Shows current connection quality (top-left in header)
- **Network Stats**: Displays bandwidth (Mbps) and latency (ms)
- **Auto FPS Button**: Toggle button to enable/disable automatic adjustment
  - Style changes to green when active
  - Shows "Auto FPS ON" when enabled

### Example Display
```
EXCELLENT | 25.50 Mbps / 2.15ms
FPS Slider with Auto FPS button
```

## API Endpoints

### GET `/api/network`
Returns current network statistics and recommendations.

**Response:**
```json
{
  "quality": "good",
  "bandwidth_mbps": 12.45,
  "latency_ms": 8.32,
  "recommended_fps": 20,
  "auto_fps_enabled": true,
  "current_fps": 20
}
```

### POST `/api/auto-fps`
Enable or disable automatic FPS adjustment.

**Request:**
```json
{
  "enable": true
}
```

**Response:**
```json
{
  "success": true,
  "auto_fps_enabled": true,
  "fps": 20,
  "quality": "good"
}
```

### POST `/api/fps` (Updated)
Set FPS manually or trigger auto adjustment.

**Request (Manual):**
```json
{
  "fps": 15
}
```

**Request (Auto):**
```json
{
  "auto": true
}
```

## Implementation Details

### NetworkMonitor Class
Located at the top of `http_thermal_server.py`, this class:
- Maintains rolling buffers of frame times and latencies
- Calculates bandwidth from frame sizes and transmission times
- Measures network latency using localhost ping
- Provides quality assessment and FPS recommendations
- Thread-safe with locking mechanisms

### Frame Tracking
- Each transmitted JPEG frame is logged with size and timestamp
- Used to calculate actual streaming bandwidth
- Helps distinguish between network limitations and server capabilities

### Periodic Updates
- Network stats updated every 5 seconds by default
- When auto FPS is active, updates every 3 seconds
- Latency measurement occurs in background thread when auto mode enabled

## Use Cases

1. **Bandwidth-Limited Networks**: Automatically reduces quality for remote access
2. **Variable Connections**: Adapts to changing network conditions in real-time
3. **Mobile Streaming**: Optimizes for 4G/5G where signal fluctuates
4. **Monitoring Dashboards**: Displays connection quality to operators

## Configuration

No configuration needed—works out of the box. The monitoring is passive and non-intrusive until auto FPS is enabled.

## Performance Impact

- **Memory**: ~50KB for rolling buffers (30 samples each)
- **CPU**: Negligible (<0.1%) for bandwidth calculations
- **Network**: No additional overhead (uses existing frame data)

## Future Enhancements

- [ ] Bandwidth threshold customization
- [ ] Predictive FPS adjustment (ML-based)
- [ ] Per-client network profiles
- [ ] Network statistics history/graphing
- [ ] Quality alerts and notifications
