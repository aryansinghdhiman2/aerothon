from flask import Flask
import socketio
from flask_socketio import SocketIO
import time
from drone_helper import connect_to_drone, Vehicle, goto_center, align_at_center, move_to_center_image_coords_with_custom_loc, drop_and_return_to_15, AUTO, configure_pid, move_to_center, getCurrentLocation, moveToAlt, get_distance_metres, LocationGlobal, drop_and_return_to_15, GUIDED, goto_center_body_ned


sio = socketio.Client()
sio.connect("http://172.24.240.1:5001")


vehicle: Vehicle = connect_to_drone("tcp:172.24.240.1:5763")
# vehicle: Vehicle = connect_to_drone("udpout:10.42.0.1:11000")

app = Flask(__name__)
socketio = SocketIO(app)

client_channel_str: str = "requested_alignment_state"

controller_15 = configure_pid(
    ((-0.0000125390625)*(1.3)), ((-0.0000125390625)*(1.3)), 0.6, 0.6)


@socketio.on("drone_data")
def handle_my_custom_event(json):
    print(str(json))


@socketio.on("alignment")
def handle_first_alignment(args):
    if (vehicle.mode == GUIDED or vehicle.mode == AUTO):
        lat, lon, alt, heading = args["location"]
        center: list[int] = args["center"]
        state: int = args["alignment_state"]
        target_or_shape: str = args["target_or_shape"]
        print(f"args: {args}")
        if state == 0:
            print("state 0")
            vehicle.mode = GUIDED
            vehicle.wait_for_mode(GUIDED)
            vehicle.parameters['WPNAV_SPEED'] = 100
            vehicle.commands.next = vehicle.commands.next - 1
            vehicle.mode = AUTO
            vehicle.wait_for_mode(AUTO)
            socketio.emit(client_channel_str, 1)
        elif state == 1:
            sio.emit("alignment", {
                     "location": args["location"], "center": args["center"]})


if __name__ == "__main__":
    socketio.run(app)
