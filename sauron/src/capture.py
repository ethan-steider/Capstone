"""Capture controller abstractions for SAURON cameras."""
from __future__ import annotations

import os
import threading
import time
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, Iterable, List, Optional, Tuple

import cv2

import camapi


@dataclass
class CameraConfig:
    """Configuration describing how to interact with a single camera feed."""

    camera_id: int
    save_path: str
    label: str


class CameraController:
    """Owns camera devices and provides capture/save helpers for the UI layer.

    The controller abstracts away three responsibilities:
    1. Lazily creating `camapi.SAURON_CAM` objects for each configured sensor.
    2. Feeding a background loop that captures and persists frames at a fixed cadence.
    3. Tracking the latest in-memory frame per label so the UI can render previews.
    """

    def __init__(self, configs: Iterable[CameraConfig], auto_create_dirs: bool = True) -> None:
        original_configs: List[CameraConfig] = list(configs)
        if not original_configs:
            raise ValueError("At least one camera configuration is required")

        self._configs = [
            CameraConfig(cfg.camera_id, os.path.abspath(cfg.save_path), cfg.label)
            for cfg in original_configs
        ]

        if auto_create_dirs:
            for cfg in self._configs:
                os.makedirs(cfg.save_path, exist_ok=True)

        self._cameras = {
            cfg.label: camapi.SAURON_CAM(cfg.camera_id, cfg.save_path) for cfg in self._configs
        }
        self._lock = threading.Lock()
        self._latest_frames: Dict[str, Optional[Tuple[datetime, "cv2.Mat"]]] = {
            cfg.label: None for cfg in self._configs
        }
        self._loop_thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._interval_seconds: float = 0.0

    def start(self, interval_seconds: float = 0.0) -> None:
        """Begins continuous capture+save on a background thread.

        A zero interval translates to "as fast as possible"; otherwise the loop sleeps
        between iterations while still reacting quickly to the stop event.
        """
        if self._loop_thread and self._loop_thread.is_alive():
            return

        self._interval_seconds = max(0.0, interval_seconds)
        self._stop_event.clear()
        self._loop_thread = threading.Thread(target=self._capture_loop, daemon=True)
        self._loop_thread.start()

    def stop(self) -> None:
        """Stops the background capture loop and waits briefly for the thread to exit."""
        self._stop_event.set()
        if self._loop_thread:
            self._loop_thread.join(timeout=1.0)
            self._loop_thread = None

    def capture_once(self, save: bool = True) -> Dict[str, "cv2.Mat"]:
        """Capture frames from every device and update the latest-frame registry.

        Args:
            save: When true, persist frames to each camera's configured directory.
        """
        timestamp = datetime.now()
        frames = {}
        for label, device in self._cameras.items():
            frame = device.capture_frame()
            frames[label] = frame
            with self._lock:
                self._latest_frames[label] = (timestamp, frame.copy())

        if save:
            self._save_frames(timestamp, frames)

        return frames

    def get_latest_frame(self, label: str) -> Optional[Tuple[datetime, "cv2.Mat"]]:
        """Return the most recently captured frame for a camera label (if any)."""
        with self._lock:
            return self._latest_frames.get(label)

    def release(self) -> None:
        """Release camera hardware handles and stop any active capture loop."""
        self.stop()
        for device in self._cameras.values():
            device.release()

    def _capture_loop(self) -> None:
        """Internal worker: capture-save loop that respects the configured interval."""
        while not self._stop_event.is_set():
            try:
                self.capture_once(save=True)
            except Exception:
                # Keep looping but respect stop event; surface errors to caller
                self._stop_event.set()
                raise

            if self._interval_seconds > 0:
                end_time = time.time() + self._interval_seconds
                while time.time() < end_time:
                    if self._stop_event.wait(timeout=0.05):
                        return
            else:
                # Yield so UI thread stays responsive even with zero interval
                if self._stop_event.wait(timeout=0.01):
                    return

    def _save_frames(self, timestamp: datetime, frames: Dict[str, "cv2.Mat"]) -> None:
        """Write each captured frame to disk using a precise timestamped filename."""
        timestamp_str = timestamp.strftime("%Y%m%d_%H-%M-%S-%f")
        for cfg in self._configs:
            frame = frames.get(cfg.label)
            if frame is None:
                continue
            filename = f"{timestamp_str}_{cfg.label}.png"
            path = os.path.join(cfg.save_path, filename)
            cv2.imwrite(path, frame)


__all__ = ["CameraConfig", "CameraController"]
