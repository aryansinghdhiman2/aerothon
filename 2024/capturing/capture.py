import cv2
import os

# Create the directory if it doesn't exist
output_dir = 'trial_131024_(2)'
os.makedirs(output_dir, exist_ok=True)

# Open the video stream
stream_url = 'http://10.42.0.1:8000/stream.mjpg'
cap = cv2.VideoCapture(stream_url)

if not cap.isOpened():
	print("Error: Could not open video stream.")
	exit()

frame_count = 0

try:
	while True:
		ret, frame = cap.read()
		if not ret:
			print("Error: Could not read frame.")
			break
		frame_filename = os.path.join(output_dir, f'frame_{frame_count:04d}.jpg')
		if (frame_count%6 != 0):
			frame_count += 1
			continue
		cv2.imwrite(frame_filename, frame)
		frame_count += 1
		cv2.imshow('Frame', frame)
		if cv2.waitKey(1) & 0xFF == ord('q'):
			break
finally:
	# Release the capture and close any OpenCV windows
	cap.release()
	cv2.destroyAllWindows()

print(f"Frames saved: {frame_count}")