import io
import logging
from flask import Flask, Response, render_template_string
from threading import Condition
from picamera2 import Picamera2
from picamera2.encoders import JpegEncoder
from picamera2.outputs import FileOutput

# HTML page template
PAGE = """\
<html>
<head>
<title>picamera2 MJPEG streaming demo</title>
</head>
<body>
<h1>Picamera2 Streaming </h1>
<img src="{{ url_for('stream') }}" width="640" height="480" />
</body>
</html>
"""

# Flask app initialization
app = Flask(__name__)

# Class to handle streaming output
class StreamingOutput(io.BufferedIOBase):
    def __init__(self):
        self.frame = None
        self.condition = Condition()

    def write(self, buf):
        with self.condition:
            self.frame = buf
            self.condition.notify_all()

# Route for the index page
@app.route('/')
def index():
    return render_template_string(PAGE)

# Route for the MJPEG stream
@app.route('/stream.mjpg')
def stream():
    def generate():
        while True:
            with output.condition:
                output.condition.wait()
                frame = output.frame
            yield (b'--FRAME\r\n'
                   b'Content-Type: image/jpeg\r\n'
                   b'Content-Length: ' + f"{len(frame)}".encode() + b'\r\n'
                   b'\r\n' + frame + b'\r\n')

    return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=FRAME')

# Initialize and configure Picamera2
picam2 = Picamera2()
picam2.configure(picam2.create_video_configuration(main={"size": (640, 480)}))
output = StreamingOutput()
picam2.start_recording(JpegEncoder(), FileOutput(output))

# Start the Flask application
if __name__ == '__main__':
    try:
        app.run(host='0.0.0.0', port=8000, threaded=True)
    finally:
        picam2.stop_recording()
