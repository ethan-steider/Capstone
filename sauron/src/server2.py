import cv2, time, threading, queue, os
from flask import Flask, Response, jsonify, render_template_string, request

# ---------- Config ----------
DEVICE = "/dev/video0"
WIDTH, HEIGHT = 640, 480
PREVIEW_FPS = 12          # browser preview rate
RECORD_FPS  = 30.0        # set 25.0 for PAL
JPEG_Q      = 60

app = Flask(__name__)

# ---------- Shared state ----------
cap = None
cap_open = False
running = False               # controls capture thread lifetime
latest = None                 # most recent frame (for preview)
last_frame_ts = 0.0
frame_cv = threading.Condition()

recording = False
record_lock = threading.Lock()
rec_thread = None
rec_q = queue.Queue(maxsize=50)   # bounded; drops oldest on overflow
writer = None

# ---------- Capture loop ----------
def open_cam():
    global cap, cap_open
    cap = cv2.VideoCapture(DEVICE)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, HEIGHT)
    cap_open = cap.isOpened()
    return cap_open

def close_cam():
    global cap, cap_open
    if cap is not None:
        try: cap.release()
        except: pass
    cap = None
    cap_open = False

def capture_loop():
    global latest, last_frame_ts, running
    while running and cap_open:
        ok, frame = cap.read()
        if not ok:
            time.sleep(0.01); continue
        last_frame_ts = time.time()
        # update latest for preview
        with frame_cv:
            latest = frame
        # push to recorder queue (non-blocking; drop if full)
        try:
            rec_q.put_nowait(frame)
        except queue.Full:
            try: rec_q.get_nowait()  # drop oldest
            except queue.Empty: pass
            try: rec_q.put_nowait(frame)
            except: pass

# ---------- Recorder thread ----------
def recorder_loop(path, fps, size):
    global writer
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # small CPU; widely playable
    writer = cv2.VideoWriter(path, fourcc, fps, size)
    if not writer.isOpened():
        writer = None
        return

    while True:
        # check stop signal frequently without busy-waiting
        with record_lock:
            if not recording: break
        try:
            frame = rec_q.get(timeout=0.2)
        except queue.Empty:
            continue
        # Defensive: writer could be closed during stop race (we hold lock above)
        writer.write(frame)

    # flush/close
    if writer is not None:
        writer.release()
        writer = None

# ---------- Web UI ----------
INDEX_HTML = """
<!doctype html>
<meta charset="utf-8"/>
<title>Drone Camera</title>
<style>
  body{font-family:system-ui,Arial;margin:16px}
  img{border:1px solid #ccc;border-radius:8px;max-width:100%}
  .row{display:flex;gap:8px;align-items:center;margin:8px 0;flex-wrap:wrap}
  button{padding:8px 12px;border-radius:8px;border:1px solid #888;cursor:pointer}
  .pill{display:inline-block;padding:4px 10px;border-radius:999px;background:#eee}
  .on{background:#c8f7c5} .off{background:#ffd3d3}
</style>
<h2>Drone Camera</h2>
<div class="row">
  <button onclick="startRec()">Start Recording</button>
  <button onclick="stopRec()">Stop Recording</button>
  <button onclick="restartFeed()">Restart Video Feed</button>
  <span id="rec" class="pill off">REC: OFF</span>
  <span id="msg"></span>
</div>
<img id="preview" src="/stream.mjpg" />
<script>
async function refresh(){
  try{
    const r = await fetch('/status', {cache:'no-store'});
    const j = await r.json();
    document.getElementById('rec').textContent = j.recording ? 'REC: ON' : 'REC: OFF';
    document.getElementById('rec').className = 'pill ' + (j.recording ? 'on' : 'off');
    document.getElementById('msg').textContent =
      (j.cap_open ? 'cam OK' : 'cam CLOSED') + ' | last frame ' + j.last_frame_age_ms + ' ms';
  }catch(e){ document.getElementById('msg').textContent='offline'; }
}
async function startRec(){
  const r = await fetch('/start_recording', {method:'POST'});
  const j = await r.json(); document.getElementById('msg').textContent = j.msg; refresh();
}
async function stopRec(){
  const r = await fetch('/stop_recording', {method:'POST'});
  const j = await r.json(); document.getElementById('msg').textContent = j.msg; refresh();
}
async function restartFeed(){
  const r = await fetch('/restart_feed', {method:'POST'});
  const j = await r.json(); document.getElementById('msg').textContent = j.msg;
  // poke the <img> to reconnect
  const img = document.getElementById('preview');
  const src = img.src.split('?')[0];
  img.src = src + '?t=' + Date.now();
  setTimeout(refresh, 500);
}
setInterval(refresh, 2000); refresh();
</script>
"""

@app.route("/")
def index():
    return render_template_string(INDEX_HTML)

@app.route("/stream.mjpg")
def stream():
    def gen():
        period = 1.0 / max(1, PREVIEW_FPS)
        while True:
            frame = None
            with frame_cv:
                frame = latest.copy() if latest is not None else None
            if frame is None:
                time.sleep(0.05); continue
            ok, jpg = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_Q])
            if not ok:
                time.sleep(0.01); continue
            yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" +
                   jpg.tobytes() + b"\r\n")
            time.sleep(period)
    return Response(gen(), mimetype="multipart/x-mixed-replace; boundary=frame")

@app.route("/start_recording", methods=["POST"])
def start_recording():
    global recording, rec_thread
    if not cap_open:
        return jsonify(ok=False, msg="Camera not open")
    with record_lock:
        if recording:
            return jsonify(ok=True, msg="Already recording")
        # clear old frames so we start fresh
        while not rec_q.empty():
            try: rec_q.get_nowait()
            except: break
        ts = time.strftime("%Y%m%d_%H%M%S")
        path = f"/home/sauron/record_{ts}.mp4"
        recording = True
        rec_thread = threading.Thread(target=recorder_loop,
                                      args=(path, RECORD_FPS, (WIDTH, HEIGHT)),
                                      daemon=True)
        rec_thread.start()
        return jsonify(ok=True, msg=f"Recording started: {path}")

@app.route("/stop_recording", methods=["POST"])
def stop_recording():
    global recording, rec_thread
    with record_lock:
        if not recording:
            return jsonify(ok=True, msg="Not recording")
        recording = False
    # Join outside lock to avoid deadlock
    if rec_thread is not None:
        rec_thread.join(timeout=2.0)
        rec_thread = None
    return jsonify(ok=True, msg="Recording stopped")

@app.route("/restart_feed", methods=["POST"])
def restart_feed():
    global running
    # Stop capture
    running = False
    time.sleep(0.1)
    close_cam()
    # Start capture
    ok = open_cam()
    if not ok:
        return jsonify(ok=False, msg="Failed to reopen camera")
    # new capture thread
    running = True
    threading.Thread(target=capture_loop, daemon=True).start()
    return jsonify(ok=True, msg="Feed restarted")

@app.route("/status")
def status():
    age_ms = int((time.time() - last_frame_ts)*1000) if last_frame_ts else -1
    return jsonify(ok=True, cap_open=cap_open, recording=recording, last_frame_age_ms=age_ms)

# ---------- Boot ----------
if __name__ == "__main__":
    try:
        if open_cam():
            running = True
            threading.Thread(target=capture_loop, daemon=True).start()
        app.run(host="0.0.0.0", port=8080, threaded=True)
    finally:
        running = False
        time.sleep(0.05)
        close_cam()
        with record_lock:
            recording = False
        if writer: 
            try: writer.release()
            except: pass