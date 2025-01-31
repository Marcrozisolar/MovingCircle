import cv2
import numpy as np

def detect_moving_circle(video_path):
    # Open the video file
    cap = cv2.VideoCapture(video_path)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Define the rotation angle
    angle = 270  # Rotate 90 degrees clockwise

    # Create a VideoWriter object to save the rotated video
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter('rotated_video.mp4', fourcc, 20.0, (height, width))


    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Convert the frame to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        blurred = cv2.GaussianBlur(gray, (9, 9), 2)

        # Detect circles using Hough Circle Transform
        circles = cv2.HoughCircles(
            blurred,
            cv2.HOUGH_GRADIENT,
            dp=1.2,
            minDist=100,
            param1=90,
            param2=90,
            minRadius=0,
            maxRadius=170
        )

        # If circles are detected, draw them
        if circles is not None:
            circles = np.uint16(np.around(circles))
            for i in circles[0, :]:
                center = (i[0], i[1])  # Circle center
                radius = i[2]          # Circle radius
                cv2.circle(frame, center, radius, (0, 255, 0), 2)  # Draw circle outline
                cv2.circle(frame, center, 2, (0, 0, 255), 3)       # Draw circle center

        rotation_matrix = cv2.getRotationMatrix2D((width / 2, height / 2), angle, 1)

        # Rotate the frame
        rotated_frame = cv2.warpAffine(frame, rotation_matrix, (width, height))

        # Write the rotated frame to the output video
        out.write(rotated_frame)

        # Display the frame
        cv2.imshow('Moving Circle Detection', rotated_frame)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(30) & 0xFF == ord('q'):
            break

    # Release resources
    cap.release()
    cv2.destroyAllWindows()

# Example usage
# Replace 'video.mp4' with the path to your video file
detect_moving_circle('/Users/pl261721/documents/PWP/movingCircle.mp4')
