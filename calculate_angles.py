import numpy as np

def calculate_angle(a, b, c):
    """
    Calculates the angle at point 'b' given three points a, b, and c.
    Points are passed as [x, y] lists or arrays.
    """
    a = np.array(a) # First point (e.g., Shoulder)
    b = np.array(b) # Mid point (e.g., Elbow)
    c = np.array(c) # End point (e.g., Wrist)

    # Calculate the radians using the arc-tangent function
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians * 180.0 / np.pi)
    
    # Ensure the angle is within 0-180 degrees
    if angle > 180.0:
        angle = 360 - angle
        
    return angle


def angles_from_landmarks(landmarks):
    # helper function to get coordinates

    output = {}

    def get_coords(landmark):
        return [landmark.x, landmark.y]

    # 1. ELBOW ANGLES (Shoulder -> Elbow -> Wrist)
    l_elbow_angle = calculate_angle(get_coords(landmarks[11]), get_coords(landmarks[13]), get_coords(landmarks[15]))
    r_elbow_angle = calculate_angle(get_coords(landmarks[12]), get_coords(landmarks[14]), get_coords(landmarks[16]))
    output['13'] = l_elbow_angle
    output['14'] = r_elbow_angle

    # 2. SHOULDER ANGLES (Hip -> Shoulder -> Elbow)
    l_shoulder_angle = calculate_angle(get_coords(landmarks[23]), get_coords(landmarks[11]), get_coords(landmarks[13]))
    r_shoulder_angle = calculate_angle(get_coords(landmarks[24]), get_coords(landmarks[12]), get_coords(landmarks[14]))
    output['11'] = l_shoulder_angle
    output['12'] = r_shoulder_angle

    # 3. KNEE ANGLES (Hip -> Knee -> Ankle)
    l_knee_angle = calculate_angle(get_coords(landmarks[23]), get_coords(landmarks[25]), get_coords(landmarks[27]))
    r_knee_angle = calculate_angle(get_coords(landmarks[24]), get_coords(landmarks[26]), get_coords(landmarks[28]))
    output['25'] = l_knee_angle
    output['26'] = r_knee_angle

    # 4. HIP/TORSO ANGLES (Shoulder -> Hip -> Knee)
    l_hip_angle = calculate_angle(get_coords(landmarks[11]), get_coords(landmarks[23]), get_coords(landmarks[25]))
    r_hip_angle = calculate_angle(get_coords(landmarks[12]), get_coords(landmarks[24]), get_coords(landmarks[26]))
    output['23'] = l_hip_angle
    output['24'] = r_hip_angle
    
    return output