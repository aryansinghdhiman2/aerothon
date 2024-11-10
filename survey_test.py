import cv2

# Load image and check if it loads correctly
image = cv2.imread('./Screenshot 2024-11-07 200626.png')
if image is None:
    print("Error: Image not found or could not be loaded.")
    exit()

# Resize the image
image = cv2.resize(image, (int(image.shape[1] * 0.7), int(image.shape[0] * 0.7)))

# Initialize shape counters
circles, squares, triangles = 0, 0, 0

# Define text properties
color = (255, 255, 0)
font = cv2.FONT_HERSHEY_SIMPLEX

# Convert image to grayscale and apply blur
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
gray_image = cv2.blur(gray_image, (5, 5))
cv2.imshow("Grayscale Image", gray_image)

# Apply thresholding
_, thresh_image = cv2.threshold(gray_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
cv2.imshow("Threshold Image", thresh_image)

# Find contours
contours, hierarchy = cv2.findContours(thresh_image, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
cv2.drawContours(image, contours, -1, (0, 255, 0), 3)

# Analyze contours
if hierarchy is not None:
    hierarchy = hierarchy[0]  # Flatten hierarchy

    for i, contour in enumerate(contours):
        if hierarchy[i][3] != -1:  # Check if the contour has a parent
            epsilon = 0.07 * cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, epsilon, True)

            x, y, w, h = cv2.boundingRect(contour)
            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

            if len(approx) == 3:
                triangles += 1
                cv2.putText(image, 'Triangle', (x, y - 10), font, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
            elif len(approx) == 4:
                squares += 1
                cv2.putText(image, 'Square', (x, y - 10), font, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
            else:
                circles += 1

# Display text with counts on image
text = f"Triangle count = {triangles}\nSquare count = {squares}\nCircle count = {circles}\nTotal shape count = {triangles + squares + circles}"
font_scale, thickness, line_type = 0.7, 2, cv2.LINE_AA
lines = text.split('\n')
height, width = image.shape[:2]
y0, dy = height - 20, 30

for i, line in enumerate(lines):
    y = y0 - i * dy * 2
    (text_width, text_height), baseline = cv2.getTextSize(line, font, font_scale, thickness)
    x = width - text_width - 60
    cv2.putText(image, line, (x, y), font, font_scale, color, thickness, line_type)

cv2.imshow("Shape Analysis", image)
cv2.waitKey(0)  # Wait until a key is pressed
cv2.destroyAllWindows()
