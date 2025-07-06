import cv2
import numpy as np
import mediapipe as mp

mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

def main():
    img = cv2.imread('sigma.jpg')
    if img is None:
        print("Error: 'meme.jpg' not found or failed to load.")
        return

    with mp_pose.Pose(static_image_mode=True) as pose:
        results = pose.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

    black_bg = np.zeros_like(img)

    if results.pose_landmarks:
        mp_drawing.draw_landmarks(
            black_bg,
            results.pose_landmarks,
            mp_pose.POSE_CONNECTIONS,
            landmark_drawing_spec=mp_drawing.DrawingSpec(color=(0,255,0), thickness=2, circle_radius=3),
            connection_drawing_spec=mp_drawing.DrawingSpec(color=(255,255,255), thickness=2)
        )
        cv2.imwrite('sigma_skeleton.png', black_bg)
        print("Skeleton image saved as 'h3_skeleton.png'")
    else:
        print("No pose detected in meme.webp")

if __name__ == '__main__':
    main()
