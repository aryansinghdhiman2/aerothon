from flask import Flask
from flask_socketio import SocketIO
import time
from drone_helper import connect_to_drone, Vehicle, goto_center, align_at_center, move_to_center_image_coords_with_custom_loc, drop_and_return_to_15, AUTO, configure_pid, move_to_center, getCurrentLocation, moveToAlt, get_distance_metres, LocationGlobal, drop_and_return_to_15

vehicle: Vehicle = connect_to_drone("tcp:localhost:5763")
# vehicle: Vehicle = connect_to_drone("udpout:10.42.0.1:11000")

app = Flask(__name__)
socketio = SocketIO(app)

client_channel_str: str = "requested_alignment_state"

controller_15 = configure_pid(-0.000375, -0.000375, 1, 1)


@socketio.on("drone_data")
def handle_my_custom_event(json):
    print(str(json))


@socketio.on("tester")
def test(args):
    print(args)
    print('Test')


@socketio.on("alignment")
def handle_first_alignment(args):

    lat, lon, alt, heading = args["location"]
    center: list[int] = args["center"]
    state = args["alignment_state"]
    print(args)
    if state == 0:
        print("state 0")
        goto_center(vehicle, center[0], center[1], lat, lon, 15, heading)
        time.sleep(2)
        socketio.emit(client_channel_str, 1)
        print("sent")
    elif state == 1:
        print("state 1")
        align_at_center(vehicle, center[0],
                        center[1], lat, lon, alt, heading)
        time.sleep(2)
        socketio.emit(client_channel_str, 2)
        print("sent")
    elif state == 2:
        print("State 2")
        move_to_center_image_coords_with_custom_loc(
            vehicle, center[0], center[1], lat, lon, 5, heading)
        time.sleep(2)
        drop_and_return_to_15(vehicle)
        socketio.emit(client_channel_str, 3)
        vehicle.mode = AUTO
        print("sent")


@socketio.on("pid")
def handle_pid(args):
    lat, lon, alt, heading = args["location"]
    delta_x, delta_y = args["center"]
    state = args["alignment_state"]

    if (abs(delta_x) > 25 or abs(delta_y) > 25):
        move_to_center(vehicle, controller_15, delta_x, delta_y)
    else:
        socketio.emit(client_channel_str, 2)
        time.sleep(1)
        print("going to 5 in server")
        location = getCurrentLocation(vehicle)
        moveToAlt(vehicle, location.lat, location.lon, 5)

        while (getCurrentLocation(vehicle).alt > 6):
            time.sleep(1)
        time.sleep(3)
        drop_and_return_to_15(vehicle)
        vehicle.mode = AUTO


if __name__ == "__main__":
    socketio.run(app)
