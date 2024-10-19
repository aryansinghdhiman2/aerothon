import socketio
import time

with socketio.SimpleClient() as sio:
  sio.connect('http://127.0.0.1:5000')
  for i in range(100):
    print(i)
    sio.emit('drone_data', {'foo': i})
    time.sleep(0.1)
print('COnnection Successful:', sio.sid)