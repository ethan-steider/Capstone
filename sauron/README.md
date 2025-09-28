# SAURON Capture Toolkit

AutoSAURON provides a desktop control panel for collecting synchronized frames from
our LWIR and RGB camera pair while also issuing low level commands to the
HM-TM5X thermal module. The application is written with Tkinter so it runs out of
the box on the lab laptops without extra GUI toolkits.

## Key Components
- `sauron/src/autosauron.py` – Tkinter UI that wires together capture control,
  live previews, and the thermal command console. The UI keeps the cameras and
  serial line responsive by using timers plus short-lived background threads.
- `sauron/src/capture.py` – Thread-aware camera controller that wraps
  `camapi.SAURON_CAM`. It handles directory creation, continuously saves frames
  with timestamped filenames, and exposes the latest frame for preview widgets.
- `sauron/src/thermal_controller.py` – Safe bridge to the HM-TM5X thermal module
  built on pyserial. It normalises command packets, manages connection state, and
  protects the serial object with a lock so UI callbacks stay thread-safe.

## Requirements
- Python 3.10+
- OpenCV (`pip install opencv-python`)
- Pillow for higher quality previews (`pip install pillow`)
- pyserial to enable the thermal command console (`pip install pyserial`)
- Local camera drivers plus the proprietary `camapi` Python bindings supplied by
  the SAURON platform team

## Running the App
1. Create a virtual environment and install the requirements listed above.
2. From the repository root run `python -m sauron.src.autosauron`.
3. Choose a capture interval (seconds) and press **Start Capture** to begin the
   continuous loop, or **Capture Snapshot** for a single save.
4. Use the thermal panel to refresh serial ports, connect to the HM-TM5X module,
   and send read/write commands. Responses appear in the console pane so command
   payloads are easy to debug.

Captured frames are saved inside `lwir_frames/` and `rgb_frames/` (relative to
where you launched the program) using nanosecond timestamps and the camera label
in the filename.

## Development Notes
- The capture module exposes `CameraController` which the UI instantiates once.
  It starts a daemon thread when `start_capture` is invoked and keeps the most
  recent frame in memory for UI previews.
- The thermal controller exposes a simple API (`connect`, `close`, `send_command`)
  so additional UI tooling or scripts can reuse it without importing any Tk code.
- All long-running or blocking operations run on background threads and marshal
  results back to Tk using `root.after`, keeping the window responsive during
  active capture sessions.

## Troubleshooting
- If previews appear blank, confirm the `camapi` bindings detect your devices and
  the storage directories are writable.
- When the thermal panel shows *pyserial is not available*, install it into your
  active environment and restart the application.
- Use the **Refresh** button in the thermal panel whenever you plug or unplug
  USB serial adapters; the dropdown repopulates with the latest list of ports.
