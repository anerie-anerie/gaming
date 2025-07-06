from flask import Flask, request, jsonify
import cv2
import numpy as np
import mediapipe as mp
import base64
from flask_cors import CORS
import os

app = Flask(__name__)
CORS(app)

mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

def decode_base64_image(data):
    if ',' in data:
        _, base64_data = data.split(',', 1)
    else:
        base64_data = data
    image_bytes = base64.b64decode(base64_data)
    nparr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    return img

def get_skeleton(img):
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
        return black_bg
    else:
        return None

def calculate_similarity(img1, img2):
    img1_gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    img2_gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    img1_gray = cv2.resize(img1_gray, (300, 300))
    img2_gray = cv2.resize(img2_gray, (300, 300))

    img1_norm = img1_gray / 255.0
    img2_norm = img2_gray / 255.0

    mse = np.mean((img1_norm - img2_norm) ** 2)

    scale_factor = 17  # tweakable
    similarity = max(0.0, 1 - mse * scale_factor)

    return round(similarity, 3)

@app.route('/compare', methods=['POST'])
def compare():
    data = request.json
    if 'image' not in data or 'meme_name' not in data:
        return jsonify({'error': 'Missing image or meme_name'}), 400

    user_img = decode_base64_image(data['image'])
    meme_name = data['meme_name']
    
    # Load correct skeleton
    skeleton_path = os.path.join("skeletons", f"{meme_name}_skeleton.png")
    if not os.path.exists(skeleton_path):
        return jsonify({'error': f'Skeleton file {meme_name}_skeleton.png not found'}), 404

    meme_skeleton = cv2.imread(skeleton_path)
    if meme_skeleton is None:
        return jsonify({'error': 'Could not read meme skeleton image'}), 500

    user_skel = get_skeleton(user_img)
    if user_skel is None:
        return jsonify({'error': 'No pose detected in user image'}), 400

    similarity = calculate_similarity(user_skel, meme_skeleton)
    return jsonify({'similarity': similarity})

if __name__ == '__main__':
    app.run(debug=True)
