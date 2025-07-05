import cv2
import mediapipe as mp
import numpy as np

mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

# Extract pose landmarks using MediaPipe
def extract_pose_landmarks(image):
    pose = mp_pose.Pose(static_image_mode=True)
    results = pose.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    if results.pose_landmarks:
        return results.pose_landmarks
    return None

# Draw skeleton on the image
def draw_skeleton(image, pose_landmarks):
    mp_drawing.draw_landmarks(
        image,
        pose_landmarks,
        mp_pose.POSE_CONNECTIONS,
        landmark_drawing_spec=mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=3),
        connection_drawing_spec=mp_drawing.DrawingSpec(color=(255, 255, 255), thickness=2)
    )
    return image

# Compare two poses and return a similarity score
def get_pose_similarity(user_image, meme_image):
    user_landmarks = extract_pose_landmarks(user_image)
    meme_landmarks = extract_pose_landmarks(meme_image)

    if not user_landmarks or not meme_landmarks:
        return 0.0

    # Convert landmarks to numpy arrays
    def landmarks_to_array(landmarks):
        return np.array([[lm.x, lm.y, lm.z] for lm in landmarks.landmark])

    user_array = landmarks_to_array(user_landmarks)
    meme_array = landmarks_to_array(meme_landmarks)

    # Normalize both sets (centered and scaled)
    def normalize(arr):
        mean = np.mean(arr, axis=0)
        arr -= mean
        max_dist = np.max(np.linalg.norm(arr, axis=1))
        return arr / max_dist if max_dist != 0 else arr

    user_array = normalize(user_array)
    meme_array = normalize(meme_array)

    # Resize to same length (just in case)
    min_len = min(len(user_array), len(meme_array))
    user_array = user_array[:min_len]
    meme_array = meme_array[:min_len]

    # Mean squared error similarity (lower is more similar)
    diff = user_array - meme_array
    mse = np.mean(np.square(diff))
    similarity = max(0.0, 100 - mse * 10)  # Scale for easier interpretation

    return round(similarity, 2)