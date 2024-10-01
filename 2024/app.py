from dronekit import connect,VehicleMode,LocationGlobalRelative,mavutil
import time
from flask import Flask,request

GUIDED = VehicleMode("GUIDED")
AUTO = VehicleMode("AUTO")
RTL = VehicleMode('RTL')

app = Flask(__name__)

@app.get("/mode/")
def getMode():
    return {
        "mode" : vehicle.mode.name
    }, 200

@app.post("/mode/auto")
def changeToAUTO():
    vehicle.mode = AUTO
    return "Mode Changed to AUTO", 200

@app.post("/mode/guided")
def changeToGUIDED():
    vehicle.mode = GUIDED
    return "Mode Changed to AUTO", 200

@app.post("/arm/")
def arm():
    vehicle.armed = True

@app.get("/arm/")
def getArmStatus():
    return {
        "armed" : vehicle.armed 
    }


@app.get("/location/")
def getLocation():
    location =  vehicle.location.global_relative_frame
    return {
        "lat":location.lat,
        "lon":location.lon,
        "alt":location.alt
    }

@app.post("/location")
def gotoLocation():
    location_raw = request.get_json()
    location = LocationGlobalRelative(location_raw["lat"],location_raw["lon"],location_raw["alt"])
    vehicle.simple_goto(location)

    return "Command Sent"

@app.post("/alt/")
def moveToAlt():
    location_raw = request.get_json()
    alt = location_raw["alt"]
    lat = location_raw["lat"]
    lon = location_raw["lon"]

    print(f"Coming down {alt}")
    msg = vehicle.message_factory.set_position_target_global_int_encode(
    0,
    0, 0,
    mavutil.mavlink.MAV_FRAME_GLOBAL_RELATIVE_ALT,  #relative to drone heading pos relative to EKF origin
    0b0000110111111000,
    int(lat * 10**7), int(lon * 10**7), alt,
    0, 0,0,
    0, 0, 0,
    0, 0)
    vehicle.send_mavlink(msg)

if(__name__=="__main__"):
    # vehicle = connect('/dev/ttyACM0',baud=5760)
    vehicle = connect('tcp:localhost:5763',baud=5760)
    vehicle.wait_ready(True,timeout=300)
    try:
        app.run(host='0.0.0.0', port=8000, threaded=True)
    finally:
        vehicle.close()