import cv2
import mediapipe as mp
import numpy as np
from calculate_angles import angles_from_landmarks
from pathlib import Path


# --- CONFIGURATION ---
INPUT_PATH = "demo_vids/babyboo.mp4"
OUTPUT_PATH = "demo_vids/processed_babyboo.mp4"

if Path(OUTPUT_PATH).is_file():
    exit(f"Output file {OUTPUT_PATH} already exists.")
# ---------------------

# 1. Initialize MediaPipe
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5, model_complexity=1)
mp_drawing = mp.solutions.drawing_utils

# 2. Setup Video Capture
cap = cv2.VideoCapture(INPUT_PATH)

# Get video properties for the writer
fps = int(cap.get(cv2.CAP_PROP_FPS))
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

# 3. Setup Video Writer
# 'mp4v' is a standard codec for .mp4 files
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(OUTPUT_PATH, fourcc, fps, (width, height))

print(f"Processing video: {INPUT_PATH}")
print(f"Total frames to process: {total_frames}")

current_frame = 0

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break

    # Process Pose
    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(image_rgb)
    
    # Draw on the original frame (not the RGB copy)
    if results.pose_landmarks:
        # Draw the skeleton
        mp_drawing.draw_landmarks(
            frame, 
            results.pose_landmarks, 
            mp_pose.POSE_CONNECTIONS,
            mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=3, circle_radius=3), # Green lines
            mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2, circle_radius=2)  # Red joints
        )
        
        # Calculate and Draw Angles
        landmarks = results.pose_landmarks.landmark
        angles = angles_from_landmarks(landmarks)
        
        for joint_id, angle in angles.items():
            joint_idx = int(joint_id)
            if joint_idx < len(landmarks):
                # Convert normalized coordinates to pixel coordinates
                x = int(landmarks[joint_idx].x * width)
                y = int(landmarks[joint_idx].y * height)
                
                # Draw angle text
                cv2.putText(frame, str(int(angle)), (x + 15, y - 15), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 3, cv2.LINE_AA)

    # Write the frame to the output file
    out.write(frame)
    
    # Progress indicator
    current_frame += 1
    if current_frame % 30 == 0:
        print(f"Progress: {current_frame}/{total_frames} frames completed...")

# Cleanup
cap.release()
out.release()
cv2.destroyAllWindows()
print(f"Finished! Processed video saved to: {OUTPUT_PATH}")