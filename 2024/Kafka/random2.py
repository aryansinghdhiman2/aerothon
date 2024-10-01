import cv2

# Load your image
image = cv2.imread('../Screenshot (760).png')

# Get the image size (frame dimensions)
height, width, layers = image.shape

# Define video properties: output filename, codec, FPS, and frame size
fps = 30  # Frames per second
video_length_in_seconds = 60  # 1-minute video
total_frames = fps * video_length_in_seconds  # Total frames needed

# Initialize the video writer object
out = cv2.VideoWriter('output_video.mp4', cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

# Loop through to write the same image for each frame
for _ in range(total_frames):
    out.write(image)

# Don't forget to release the writer when done!
out.release()

print("Video created successfully! 🎉")
