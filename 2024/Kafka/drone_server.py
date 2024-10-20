from flask import Flask
from flask_socketio import SocketIO
import time
from drone_helper import connect_to_drone, Vehicle, goto_center, align_at_center, move_to_center_image_coords_with_custom_loc, drop_and_return_to_15, AUTO

# vehicle: Vehicle = connect_to_drone("tcp:localhost:5763")
vehicle: Vehicle = connect_to_drone("udpout:10.42.0.1:11000")

app = Flask(__name__)
socketio = SocketIO(app)

client_channel_str: str = "requested_alignment_state"


@socketio.on("drone_data")
def handle_my_custom_event(json):
    print(str(json))


@socketio.on("alignment")
def handle_first_alignment(args):
    lat, lon, alt, heading = args["location"]
    center: list[int] = args["center"]
    state = args["alignment_state"]
    if state == 0:
        print("state 0")
        goto_center(vehicle, center[0], center[1], lat, lon, 15, heading)
        time.sleep(2)
        socketio.emit(client_channel_str, 1)
    elif state == 1:
        print("state 1")
        align_at_center(vehicle, center[0],
                        center[1], lat, lon, alt, heading)
        time.sleep(2)
        socketio.emit(client_channel_str, 2)
    elif state == 2:
        print("State 2")
        move_to_center_image_coords_with_custom_loc(
            vehicle, center[0], center[1], lat, lon, 15, heading)
        time.sleep(2)
        drop_and_return_to_15(vehicle)
        socketio.emit(client_channel_str, 3)
        vehicle.mode = AUTO


if __name__ == "__main__":
    socketio.run(app)
