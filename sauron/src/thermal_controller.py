"""Serial helpers for configuring the HM-TM5X-XRG/C thermal camera module."""
from __future__ import annotations

import threading
from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Tuple

try:  # pyserial is required for runtime use
    import serial  # type: ignore
    from serial.tools import list_ports  # type: ignore
except Exception:  # pragma: no cover - handled gracefully at runtime
    serial = None  # type: ignore
    list_ports = None  # type: ignore


BEGIN_BYTE = 0xF0
END_BYTE = 0xFF
HEADER_DATA_BYTES = 0x04


class SerialUnavailableError(RuntimeError):
    """Raised when pyserial is not installed in the current environment."""


class ThermalSerialControllerError(RuntimeError):
    """Raised when a serial operation fails."""


@dataclass(slots=True)
class ThermalCommand:
    """Represents a UART command targeting the HM-TM5X thermal module."""

    device_address: int
    class_address: int
    subclass_address: int
    rw_flag: int
    data: Sequence[int] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        for value in (self.device_address, self.class_address, self.subclass_address, self.rw_flag):
            _ensure_byte(value, "command header")
        for value in self.data:
            _ensure_byte(value, "command data")

    @property
    def checksum(self) -> int:
        """Checksum is the low byte of the header+payload sum."""
        total = self.device_address + self.class_address + self.subclass_address + self.rw_flag
        total += sum(self.data)
        return total & 0xFF

    @property
    def payload_length(self) -> int:
        return len(self.data)

    def to_bytes(self) -> bytes:
        """Materialise the command into the byte format expected by the module."""
        data_bytes = bytes(self.data)
        size = HEADER_DATA_BYTES + len(data_bytes)
        packet = bytearray(6 + len(data_bytes) + 2)
        packet[0] = BEGIN_BYTE
        packet[1] = size & 0xFF
        packet[2] = self.device_address
        packet[3] = self.class_address
        packet[4] = self.subclass_address
        packet[5] = self.rw_flag
        if data_bytes:
            packet[6 : 6 + len(data_bytes)] = data_bytes
        packet[6 + len(data_bytes)] = self.checksum
        packet[7 + len(data_bytes)] = END_BYTE
        return bytes(packet)


def enumerate_serial_ports() -> List[str]:
    """Return a list of available serial port device paths."""
    if list_ports is None:
        return []
    return [port.device for port in list_ports.comports()]


class ThermalSerialController:
    """Thread-safe helper that manages the UART session with the thermal module.

    The UI layer holds a single instance of this controller and routes connect,
    disconnect, and send commands through it. A re-entrant lock protects access to
    the underlying `serial.Serial` object because Tk callbacks may run on
    different threads when background workers post results back to the GUI.
    """

    def __init__(self) -> None:
        if serial is None:
            raise SerialUnavailableError(
                "pyserial is required to control the thermal camera. pip install pyserial"
            )
        self._lock = threading.Lock()
        self._serial: Optional[serial.Serial] = None
        self._port: Optional[str] = None
        self._baudrate: Optional[int] = None
        self._timeout: float = 0.1

    @property
    def is_connected(self) -> bool:
        with self._lock:
            return self._serial is not None and self._serial.is_open

    @property
    def port(self) -> Optional[str]:
        with self._lock:
            return self._port

    def connect(self, port: str, baudrate: int = 115200, timeout: float = 0.1) -> None:
        """Open (or re-open) the serial port with the supplied parameters."""
        if serial is None:  # Defensive guard even though __init__ checks
            raise SerialUnavailableError(
                "pyserial is required to control the thermal camera. pip install pyserial"
            )

        if not port:
            raise ThermalSerialControllerError("A serial port path is required")

        with self._lock:
            if self._serial and self._serial.is_open:
                same_port = self._serial.port == port
                same_baud = int(self._serial.baudrate) == int(baudrate)
                if same_port and same_baud:
                    self._serial.timeout = timeout
                    self._timeout = timeout
                    return
                self._close_locked()

            try:
                self._serial = serial.Serial(port=port, baudrate=baudrate, timeout=timeout)
            except serial.SerialException as exc:  # type: ignore[attr-defined]
                raise ThermalSerialControllerError(str(exc)) from exc

            self._port = port
            self._baudrate = int(baudrate)
            self._timeout = timeout

    def close(self) -> None:
        """Close the port if it is open, swallowing hardware errors on shutdown."""
        with self._lock:
            self._close_locked()

    def send_command(
        self,
        command: ThermalCommand,
        *,
        read_mode: str = "line",
        read_bytes: int = 64,
        read_timeout: float = 0.2,
    ) -> Tuple[bytes, bytes]:
        """Send a command and optionally read a response.

        Args:
            command: The command payload to transmit.
            read_mode: 'line' reads until newline/timeout; 'bytes' reads a fixed byte count.
            read_bytes: Number of bytes to read when in 'bytes' mode.
            read_timeout: Temporary timeout applied while waiting for a response.

        Returns:
            A tuple containing the transmitted packet and any bytes received in
            accordance with ``read_mode``.
        """
        serial_ref = self._get_serial()
        packet = command.to_bytes()

        with self._lock:
            serial_ref.reset_input_buffer()
            try:
                serial_ref.write(packet)
                serial_ref.flush()
            except serial.SerialException as exc:  # type: ignore[attr-defined]
                raise ThermalSerialControllerError(str(exc)) from exc

            response = b""
            old_timeout = serial_ref.timeout
            serial_ref.timeout = read_timeout
            try:
                if read_mode == "line":
                    response = serial_ref.readline()
                elif read_mode == "bytes":
                    if read_bytes <= 0:
                        read_bytes = 64
                    response = serial_ref.read(read_bytes)
                elif read_mode == "none":
                    response = b""
                else:
                    raise ThermalSerialControllerError(f"Unsupported read mode: {read_mode}")
            except serial.SerialException as exc:  # type: ignore[attr-defined]
                raise ThermalSerialControllerError(str(exc)) from exc
            finally:
                serial_ref.timeout = old_timeout

        return packet, response

    def _get_serial(self) -> serial.Serial:  # type: ignore[return-value]
        with self._lock:
            if self._serial is None or not self._serial.is_open:
                raise ThermalSerialControllerError("Serial port is not connected")
            return self._serial

    def _close_locked(self) -> None:
        """Close the serial object in-place; caller must hold ``self._lock``."""
        if self._serial and self._serial.is_open:
            try:
                self._serial.close()
            except Exception:
                pass
        self._serial = None
        self._port = None
        self._baudrate = None
        self._timeout = 0.1


def _ensure_byte(value: int, context: str) -> None:
    """Validate that ``value`` fits in an unsigned byte for the provided context."""
    if not 0 <= int(value) <= 0xFF:
        raise ThermalSerialControllerError(f"Value {value!r} out of byte range in {context}")
