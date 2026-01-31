const DATA_PATH = "./data";
const STATUS_URL = `${DATA_PATH}/status.json`;
const THERMAL_URL = `${DATA_PATH}/latest-thermal.jpg`;
const RGB_URL = `${DATA_PATH}/latest-rgb.jpg`;
const POLL_INTERVAL_MS = 5000;

const el = (id) => document.getElementById(id);

const formatNumber = (value, suffix = "", fallback = "--") => {
  if (value === null || value === undefined || Number.isNaN(value)) return fallback;
  return `${value}${suffix}`;
};

const timeAgo = (iso) => {
  if (!iso) return "--";
  const then = new Date(iso);
  if (Number.isNaN(then.getTime())) return iso;
  const delta = (Date.now() - then.getTime()) / 1000;
  if (delta < 60) return `${Math.floor(delta)}s ago`;
  if (delta < 3600) return `${Math.floor(delta / 60)}m ago`;
  if (delta < 86400) return `${Math.floor(delta / 3600)}h ago`;
  return `${Math.floor(delta / 86400)}d ago`;
};

async function fetchJson(url) {
  const res = await fetch(`${url}?ts=${Date.now()}`, { cache: "no-store" });
  if (!res.ok) throw new Error(`fetch failed ${res.status}`);
  return res.json();
}

function setImg(kind, url) {
  const img = el(`${kind}-img`);
  const empty = el(`${kind}-empty`);
  const meta = el(`${kind}-meta`);
  if (!url) {
    img.classList.remove("ready");
    img.src = "";
    empty.style.display = "grid";
    meta.textContent = "no frame";
    return;
  }
  const objectUrl = URL.createObjectURL(url);
  img.onload = () => {
    img.classList.add("ready");
    empty.style.display = "none";
    URL.revokeObjectURL(objectUrl);
  };
  img.onerror = () => {
    img.classList.remove("ready");
    empty.style.display = "grid";
    meta.textContent = "failed to load";
    URL.revokeObjectURL(objectUrl);
  };
  img.src = objectUrl;
  meta.textContent = "updated";
}

async function loadImage(kind, targetUrl) {
  try {
    const res = await fetch(`${targetUrl}?ts=${Date.now()}`, { cache: "no-store" });
    if (!res.ok) throw new Error(`status ${res.status}`);
    const blob = await res.blob();
    if (!blob.type.startsWith("image/")) throw new Error("not an image");
    setImg(kind, blob);
  } catch (err) {
    console.warn(`no ${kind} frame`, err);
    setImg(kind, null);
  }
}

function updateStatus(payload) {
  const status = payload.status || {};
  const network = payload.network || {};
  const palette = payload.palette || {};
  const updatedAt = payload.generated_at || payload.updated_at || null;

  el("pill-state").textContent = payload.online ? "data online" : "waiting for data";
  el("pill-state").style.borderColor = payload.online ? "rgba(56, 198, 198, 0.4)" : "var(--border)";
  el("pill-updated").textContent = updatedAt ? `Updated ${timeAgo(updatedAt)}` : "--";

  el("net-quality").textContent = network.quality || "--";
  el("net-bandwidth").textContent = formatNumber(
    network.bandwidth_mbps ? Number(network.bandwidth_mbps).toFixed(2) : null,
    " Mbps"
  );
  el("net-latency").textContent = formatNumber(
    network.latency_ms ? Number(network.latency_ms).toFixed(1) : null,
    " ms"
  );
  el("net-fps").textContent = formatNumber(network.recommended_fps, " fps");

  const rgbStatus = status.cameras?.rgb?.running ? "running" : "stopped";
  const thermalStatus = status.cameras?.thermal?.running ? "running" : "stopped";
  el("cam-rgb").textContent = rgbStatus;
  el("cam-thermal").textContent = thermalStatus;
  el("cam-fps").textContent = formatNumber(status.current_fps, " fps");

  const paletteName = palette.palette_name || status.palette || "--";
  el("cam-palette").textContent = paletteName;
}

async function loadStatus() {
  try {
    const payload = await fetchJson(STATUS_URL);
    updateStatus({ ...payload, online: true });
  } catch (err) {
    console.warn("status unavailable", err);
    updateStatus({ online: false });
  }
}

async function tick() {
  await Promise.all([
    loadStatus(),
    loadImage("thermal", THERMAL_URL),
    loadImage("rgb", RGB_URL),
  ]);
  setTimeout(tick, POLL_INTERVAL_MS);
}

document.addEventListener("DOMContentLoaded", () => {
  tick();
});
