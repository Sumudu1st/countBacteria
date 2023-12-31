import cv2


# Open the video file
video_path = "bac1.mp4"
cap = cv2.VideoCapture(video_path)

while True:
    # Read a frame from the video
    ret, frame = cap.read()

    # If the frame is not read successfully, break the loop
    if not ret:
        break

    # Convert the frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Apply thresholding
    thresh, binary = cv2.threshold(gray, 150, 200, cv2.THRESH_BINARY)

    # Find contours
    contours, hierarchy = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Draw contours on the original frame
    cv2.drawContours(frame, contours, -1, (0, 255, 0), 2)

    # Get the number of bacteria
    number_of_bacteria = len(contours)

    # Display the bacteria count on the frame
    cv2.putText(frame, f'Bacteria Count: {number_of_bacteria}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Display the frame
    cv2.imshow("Bacteria Detection", frame)

    # Break the loop if 'q' key is pressed
    if cv2.waitKey(25) & 0xFF == ord('q'):
        break

# Release the video capture object and close all windows
cap.release()
cv2.destroyAllWindows()
