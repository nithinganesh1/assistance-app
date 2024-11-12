import cv2

# Try opening the external camera at index 2
cap = cv2.VideoCapture(1)

if not cap.isOpened():
    print("Error: Could not open camera at index 2.")
    exit()

# Loop to continuously capture frames from the external camera
while True:
    ret, frame = cap.read()

    if not ret:
        print("Error: Failed to capture frame.")
        break

    # Display the frame in a window
    cv2.imshow('External Camera Feed', frame)

    # Exit the video window on pressing 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the camera and close any open windows
cap.release()
cv2.destroyAllWindows()
