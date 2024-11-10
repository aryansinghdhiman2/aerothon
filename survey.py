import cv2

capture = cv2.VideoCapture('shapes.mp4')
while True:
    ret, image = capture.read()
    if not ret:
        break
    image = cv2.resize(
        image, (int(image.shape[1]*0.7), int(image.shape[0]*0.7)))

    circles = 0
    squares = 0
    triangles = 0
    color = (255, 255, 0)
    font = cv2.FONT_HERSHEY_SIMPLEX

    # Converting the given image to grayscale
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray_image = cv2.blur(gray_image, (5, 5))

    cv2.imshow("Greyed", gray_image)
    
    # Applying thresholding to convert to binary masked image
    _, thresh_image = cv2.threshold(
        gray_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    cv2.imshow("Threshed", thresh_image)
    
    # Generating contours with tree method and relationship through hierarchy
    contours, hierarchy = cv2.findContours(
        thresh_image, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    
    cv2.drawContours(image, contours, -1, (0, 255, 0), 3)
    
    if hierarchy is not None:
        hierarchy = hierarchy[0]  # Flatten the hierarchy
        
        for i, contour in enumerate(contours):
            # Check if the contour is a child (i.e., it has a parent contour)
            if hierarchy[i][3] != -1:  # Parent != -1 means it has a parent
                epsilon = 0.07 * cv2.arcLength(contour, True)
                approx = cv2.approxPolyDP(contour, epsilon, True)

                # Get the bounding box coordinates
                x, y, w, h = cv2.boundingRect(contour)

                cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

                if len(approx) == 3:
                    triangles += 1
                    cv2.putText(image, 'Triangle', (x, y - 10), font,
                                0.5, (0, 0, 255), 1, cv2.LINE_AA)
                elif len(approx) == 4:
                    squares += 1
                    cv2.putText(image, 'Square', (x, y - 10), font,
                                0.5, (0, 0, 255), 1, cv2.LINE_AA)
                else:
                    circles += 1

    # Display the text with counts
    text = f"Triangle count = {triangles}\nSquare count = {squares}\nTotal shape count = {triangles + squares}"
    font_scale = 0.7
    color = (0, 255, 255) 
    thickness = 2
    line_type = cv2.LINE_AA

    lines = text.split('\n')

    (height, width) = image.shape[:2]

    y0, dy = height - 20, 30 

    for i, line in enumerate(lines):
        y = y0 - i * dy * 2
        (text_width, text_height), baseline = cv2.getTextSize(line, font, font_scale, thickness)
        x = width - text_width - 60
        cv2.putText(image, line, (x, y), font, font_scale, color, thickness, line_type)  

    cv2.imshow("survey", image)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

capture.release()
cv2.destroyAllWindows()
