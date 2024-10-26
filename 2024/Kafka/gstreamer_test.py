import cv2
# import sys
# Open the default camera
cam = cv2.VideoCapture("rtspsrc location=rtsp://10.42.0.1:8554/cam latency=0 protocols=tcp ! decodebin ! videoconvert ! appsink drop=1 max-buffers=5 max-bytes=1843488",cv2.CAP_GSTREAMER)

# print(cam.getBackendName())

# Get the default frame width and height
frame_width = int(cam.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cam.get(cv2.CAP_PROP_FRAME_HEIGHT))


while True:
    ret, frame = cam.read()
    # print(sys.getsizeof(frame))



    # Display the captured frame
    cv2.imshow('Camera', frame)

    # Press 'q' to exit the loop
    if cv2.waitKey(1) == ord('q'):
        break

# Release the capture and writer objects
cam.release()
cv2.destroyAllWindows()