from flask import Flask
from flask_socketio import SocketIO
import time

app = Flask(__name__)
socketio = SocketIO(app)

@socketio.on("drone_data")
def handle_my_custom_event(json):
  print(str(json))
if __name__ == "__main__":
  socketio.run(app)