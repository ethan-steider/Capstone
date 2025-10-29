import cv2, time
from flask import Flask, Response
app = Flask(__name__)

cap = cv2.VideoCapture(0)  # /dev/video0
cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))  # ask for MJPEG
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
cap.set(cv2.CAP_PROP_FPS, 5)  # low FPS preview

def gen():
    while True:
        ok, frame = cap.read()
        if not ok: time.sleep(0.05); continue
        # (optional) naive deinterlace:
        # frame = cv2.resize(frame[0::2], (frame.shape[1], frame.shape[0]))
        ret, jpg = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), 60])
        if not ret: continue
        yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' +
               jpg.tobytes() + b'\r\n')

@app.route('/stream')
def stream(): return Response(gen(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/')
def index(): return '<img src="/stream" />'
app.run(host='0.0.0.0', port=8080, threaded=True)

if __name__ == '__main__':
app.run(host='0.0.0.0', port=8080, threaded=True)