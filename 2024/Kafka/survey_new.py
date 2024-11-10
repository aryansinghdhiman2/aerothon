import cv2
import numpy as np

# Load video
capture = cv2.VideoCapture(
        "rtspsrc location=rtsp://172.24.240.1:8554/cam latency=0 protocols=tcp ! decodebin ! videoconvert ! appsink drop=1 max-buffers=5 max-bytes=1843488", cv2.CAP_GSTREAMER)

fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # 'mp4v' for MP4 output format
out = cv2.VideoWriter("saved.mp4", fourcc, 6, (640, 480))  # Save as .mp4

min_area = 900  # Adjust as needed
distance_threshold = 15  # Tolerance for centroid matching

while True:
    ret, image = capture.read()
    if not ret:
        break

    image = cv2.resize(image, (int(image.shape[1] * 0.7), int(image.shape[0] * 0.7)))
    new_image = image.copy()

    font = cv2.FONT_HERSHEY_SIMPLEX

    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred_gray_image = cv2.blur(gray_image, (7, 7))
    
    _, threshed_image = cv2.threshold(blurred_gray_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Morphological transformations
    kernel1 = np.ones((1, 1), np.uint8)
    eroded_image = cv2.erode(threshed_image, kernel1, iterations=1)
    dilated_image = cv2.dilate(eroded_image, kernel1, iterations=1)

    contours, hierarchy = cv2.findContours(dilated_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    hsv_frame = cv2.cvtColor(new_image, cv2.COLOR_BGR2HSV)
    lower_red = np.array([141, 77, 129])
    upper_red = np.array([255, 255, 255])
    mask = cv2.inRange(hsv_frame, lower_red, upper_red)
    result = cv2.bitwise_and(new_image, new_image, mask=mask)

    mask_contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    def get_centroid(contour):
        M = cv2.moments(contour)
        if M["m00"] == 0:
            return None
        return (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))

    triangles = 0
    squares = 0
    color = (0, 255, 0)

    parent_shapes = {}

    for i, contour in enumerate(contours):
        if cv2.contourArea(contour) > min_area:
            epsilon = 0.07 * cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, epsilon, True)
            shape_type = None
            if len(approx) == 3:
                shape_type = "Triangle"
            elif len(approx) == 4:
                shape_type = "Square"
            else:
                shape_type = "Circle"

            centroid = get_centroid(contour)
            if centroid is None:
                continue

            for mask_contour in mask_contours:
                if cv2.contourArea(mask_contour) > min_area:
                    mask_centroid = get_centroid(mask_contour)
                    if mask_centroid is None:
                        continue

                    distance = np.sqrt((centroid[0] - mask_centroid[0]) ** 2 + (centroid[1] - mask_centroid[1]) ** 2)
                    if distance < distance_threshold:
                        # Annotate the confirmed shape
                        if shape_type == "Triangle":
                            triangles += 1
                        elif shape_type == "Square":
                            squares += 1
                        
                        # Draw and label shape on the image
                        cv2.putText(image, shape_type, (approx[0][0][0], approx[0][0][1] - 10), font, 0.5, (0, 0, 255), 2, cv2.LINE_AA)
                        cv2.drawContours(image, [contour], -1, (0, 0, 0), 1)
                        x, y, w, h = cv2.boundingRect(contour)
                        cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)

                        # Store shape under its parent
                        parent_idx = hierarchy[0][i][3]
                        if parent_idx not in parent_shapes:
                            parent_shapes[parent_idx] = []
                        parent_shapes[parent_idx].append(shape_type)
    center = (0,0)
    for parent_idx, shapes in parent_shapes.items():
        if len(shapes) > 1:  # Only consider parents with multiple child shapes
            merged_contour = contours[parent_idx]
            cv2.drawContours(image, [merged_contour], -1, (255, 255, 0), 2) 
            (center_x, center_y), radius = cv2.minEnclosingCircle(merged_contour)
            center = (int(center_x), int(center_y))
            radius = int(radius)
            cv2.circle(image, center, radius, (255, 255, 0), 2)  # Yellow enclosing circle for parent
            cv2.circle(image, center, 5, (0, 0, 255), -1)  # Red center

    # Display the frames
    cv2.imshow("Masked Image", result)
    cv2.imshow("Survey", image)
    out.write(image)

    # Print shape counts
    print("Triangles:", triangles, "Squares:", squares, "Total:", triangles + squares, "center", center)

    # Exit on 'q' key
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
capture.release()
cv2.destroyAllWindows()
