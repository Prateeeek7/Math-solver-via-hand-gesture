import cv2

# Open webcam (0 is usually your default webcam)
cap = cv2.VideoCapture(0)

while True:
    success, frame = cap.read()  # Read a frame from the webcam
    if not success:
        break

    cv2.imshow("Webcam Feed", frame)  # Show the video frame

    # Press 'q' to quit the webcam window
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release webcam and close all windows
cap.release()
cv2.destroyAllWindows()
