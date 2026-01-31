"""
Pull a snapshot from the LAN-only http_thermal_server and write static assets
into docs/data/ for GitHub Pages. No changes to the server are required.

Usage:
  cd sauron && python3 src/github_pages_pusher.py --server http://192.168.0.10:8080

Run this on a gateway that can reach the LAN server and the internet (for pushing
to GitHub). Commit the updated docs/data files and push to publish.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional
from urllib import error, request


def _fetch_json(url: str, timeout: float = 5.0) -> Optional[Dict[str, Any]]:
    req = request.Request(url, headers={"User-Agent": "sauron-pages-pusher"})
    try:
        with request.urlopen(req, timeout=timeout) as resp:
            charset = resp.headers.get_content_charset() or "utf-8"
            raw = resp.read().decode(charset, errors="replace")
            return json.loads(raw)
    except Exception as exc:  # noqa: BLE001
        print(f"[WARN] Failed to fetch JSON from {url}: {exc}")
        return None


def _fetch_mjpeg_frame(url: str, timeout: float = 5.0, max_buffer: int = 512_000) -> Optional[bytes]:
    """
    Read a single JPEG frame from an MJPEG stream by scanning for JPEG SOI/EOI bytes.
    """
    req = request.Request(url, headers={"User-Agent": "sauron-pages-pusher"})
    start = time.time()
    try:
        with request.urlopen(req, timeout=timeout) as resp:
            # Best-effort socket timeout on the underlying socket (may not exist on all platforms)
            try:
                sock = getattr(resp, "fp", None)
                raw = getattr(sock, "raw", None) if sock else None
                if raw and hasattr(raw, "_sock"):
                    raw._sock.settimeout(timeout)
            except Exception:
                pass

            buf = b""
            while time.time() - start < timeout:
                chunk = resp.read(4096)
                if not chunk:
                    break
                buf += chunk
                soi = buf.find(b"\xff\xd8")  # Start of Image
                eoi = buf.find(b"\xff\xd9", soi + 2 if soi != -1 else 0)  # End of Image
                if soi != -1 and eoi != -1 and eoi > soi:
                    return buf[soi : eoi + 2]
                if len(buf) > max_buffer:
                    buf = buf[-max_buffer:]
    except error.URLError as exc:
        print(f"[WARN] Failed to fetch MJPEG from {url}: {exc}")
    except Exception as exc:  # noqa: BLE001
        print(f"[WARN] Unexpected MJPEG error for {url}: {exc}")
    return None


def _write_json(path: Path, data: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def _write_bytes(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)


def _cli() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Push LAN thermal snapshots into docs/data for GitHub Pages.")
    parser.add_argument("--server", default="http://localhost:8080", help="Base URL of the LAN http_thermal_server")
    parser.add_argument(
        "--output",
        default=str(Path(__file__).resolve().parents[2] / "docs" / "data"),
        help="Destination folder for static files (default: docs/data)",
    )
    parser.add_argument("--timeout", type=float, default=5.0, help="Timeout (seconds) for HTTP requests")
    parser.add_argument("--skip-rgb", action="store_true", help="Skip RGB frame capture")
    parser.add_argument("--skip-thermal", action="store_true", help="Skip thermal frame capture")
    parser.add_argument("--no-palette", action="store_true", help="Skip palette read")
    return parser.parse_args()


def main() -> int:
    args = _cli()
    base = args.server.rstrip("/")
    output_dir = Path(args.output)

    print(f"[info] Reading from {base} -> writing to {output_dir}")

    status = _fetch_json(f"{base}/api/status", timeout=args.timeout) or {}
    network = _fetch_json(f"{base}/api/network", timeout=args.timeout) or {}
    palette = {} if args.no_palette else (_fetch_json(f"{base}/api/thermal/palette", timeout=args.timeout) or {})

    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "source_server": base,
        "status": status,
        "network": network,
        "palette": palette,
    }
    _write_json(output_dir / "status.json", payload)
    print("[ok] Wrote status.json")

    if not args.skip_thermal:
        thermal_frame = _fetch_mjpeg_frame(f"{base}/thermal", timeout=args.timeout)
        if thermal_frame:
            _write_bytes(output_dir / "latest-thermal.jpg", thermal_frame)
            print("[ok] Wrote latest-thermal.jpg")
        else:
            print("[warn] No thermal frame captured")

    if not args.skip_rgb:
        rgb_frame = _fetch_mjpeg_frame(f"{base}/stream", timeout=args.timeout)
        if rgb_frame:
            _write_bytes(output_dir / "latest-rgb.jpg", rgb_frame)
            print("[ok] Wrote latest-rgb.jpg")
        else:
            print("[warn] No RGB frame captured")

    return 0


if __name__ == "__main__":
    sys.exit(main())
