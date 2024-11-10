# import cv2
# cap = cv2.VideoCapture(
#       "rtspsrc location=rtsp://172.24.240.1:8554/cam latency=0 protocols=tcp ! decodebin ! videoconvert ! appsink drop=1 max-buffers=5 max-bytes=1843488", cv2.CAP_GSTREAMER)
# if not cap.isOpened():
#     print("Error: Unable to open RTSP stream")
#     exit()
# frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
# frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# output_file = "./TestResults/manual_2.avi"
# fourcc = cv2.VideoWriter_fourcc(*"XVID")
# out = cv2.VideoWriter(output_file, fourcc, 20.0, (frame_width, frame_height))

# while True:
#     ret, frame = cap.read()
#     if not ret:
#         print("Error: Failed to retrieve frame. Exiting.")
#         break

#     # Write the frame to the output video file
#     out.write(frame)

#     # Optional: Display the frame
#     cv2.imshow("RTSP Stream", frame)
#     if cv2.waitKey(1) & 0xFF == ord("q"):
#         break

# # Release resources
# cap.release()
# out.release()
# cv2.destroyAllWindows()


