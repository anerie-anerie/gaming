import cv2
import mediapipe as mp
import numpy as np

mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

def extract_pose_landmarks(image):
    with mp_pose.Pose(static_image_mode=True) as pose:
        results = pose.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        if results.pose_landmarks:
            return results.pose_landmarks
        return None

def draw_skeleton(image, pose_landmarks):
    mp_drawing.draw_landmarks(
        image,
        pose_landmarks,
        mp_pose.POSE_CONNECTIONS,
        landmark_drawing_spec=mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=3),
        connection_drawing_spec=mp_drawing.DrawingSpec(color=(255, 255, 255), thickness=2)
    )
    return image

def get_angle(a, b, c):
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)

    ba = a - b
    bc = c - b

    cosine = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-6)
    angle = np.arccos(np.clip(cosine, -1.0, 1.0))
    return np.degrees(angle)

def extract_angles(landmarks):
    points = [(lm.x, lm.y) for lm in landmarks.landmark]
    angles = []

    def angle_between(i, j, k):
        angles.append(get_angle(points[i], points[j], points[k]))

    # Elbow angles
    angle_between(12, 14, 16)  # Right arm
    angle_between(11, 13, 15)  # Left arm

    # Shoulder-hip-knee (upper body lean)
    angle_between(24, 12, 14)  # Right shoulder
    angle_between(23, 11, 13)  # Left shoulder

    # Knee angles
    angle_between(24, 26, 28)  # Right knee
    angle_between(23, 25, 27)  # Left knee

    return np.array(angles)

def get_pose_similarity(user_image, meme_image):
    user_landmarks = extract_pose_landmarks(user_image)
    meme_landmarks = extract_pose_landmarks(meme_image)

    if not user_landmarks or not meme_landmarks:
        return 0.0

    user_angles = extract_angles(user_landmarks)
    meme_angles = extract_angles(meme_landmarks)

    if len(user_angles) != len(meme_angles):
        return 0.0

    diff = np.abs(user_angles - meme_angles)
    avg_diff = np.mean(diff)

    similarity = max(0.0, 100 - avg_diff)  # Lower avg_diff means higher similarity
    return round(similarity, 2)