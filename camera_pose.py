import cv2
import mediapipe as mp

# Initialize MediaPipe Pose and Drawing utilities
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)

# We'll keep only the body: 11-32 (shoulders, arms, torso, legs)
BODY_LANDMARKS = list(range(11, 33))

# Open the webcam (0 is usually the default camera)
cap = cv2.VideoCapture(0)

print("Starting pose tracking... Press 'q' to quit.")

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        print("Ignoring empty camera frame.")
        continue

    # 1. Flip the image horizontally for a selfie-view display
    # 2. Convert the BGR image to RGB (MediaPipe requires RGB)
    image = cv2.cvtColor(cv2.flip(frame, 1), cv2.COLOR_BGR2RGB)
    
    # To improve performance, optionally mark the image as not writeable
    image.flags.writeable = False
    results = pose.process(image)

    # Draw the pose landmarks on the frame
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR) # Convert back to BGR for OpenCV
    
    if results.pose_landmarks:
        mp_drawing.draw_landmarks(
            image, 
            results.pose_landmarks, 
            mp_pose.POSE_CONNECTIONS,
            mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
            mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2, circle_radius=1)
        )

    # Display the resulting frame
    cv2.imshow('Real-Time Pose Tracking', image)

    # Break the loop when 'q' is pressed
    if cv2.waitKey(5) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()