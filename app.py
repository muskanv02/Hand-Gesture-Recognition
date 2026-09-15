from flask import Flask, render_template, jsonify, request
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import mediapipe as mp
import joblib
import numpy as np
import cv2
import os
import base64

app = Flask(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "models", "gesture_model.pkl")
LANDMARKER_PATH = os.path.join(BASE_DIR, "models", "hand_landmarker.task")

model = None
landmarker = None

GESTURE_EMOJIS = {
    "Open_Palm": "🖐️",
    "Fist": "✊",
    "Thumbs_Up": "👍",
    "Thumbs_Down": "👎",
    "Peace": "✌️",
    "OK": "👌",
    "Pointing": "☝️",
    "Love": "🤟",
    "Rock": "🤘",
    "One_Finger": "☝️",
}

def load_models():
    global model, landmarker

    try:
        if os.path.exists(MODEL_PATH):
            model = joblib.load(MODEL_PATH)
            print("Gesture model loaded successfully.")
        else:
            print("Gesture model not found:", MODEL_PATH)
    except Exception as e:
        print("Error loading gesture model:", e)

    try:
        if os.path.exists(LANDMARKER_PATH):
            base_options = python.BaseOptions(model_asset_path=LANDMARKER_PATH)
            options = vision.HandLandmarkerOptions(
                base_options=base_options,
                num_hands=1,
                min_hand_detection_confidence=0.3,
                min_hand_presence_confidence=0.3,
                min_tracking_confidence=0.3,
            )
            landmarker = vision.HandLandmarker.create_from_options(options)
            print("MediaPipe Hand Landmarker loaded successfully.")
        else:
            print("hand_landmarker.task not found:", LANDMARKER_PATH)
    except Exception as e:
        print("Error loading MediaPipe:", e)

load_models()

def extract_landmarks(hand_landmarks):
    base_x = hand_landmarks[0].x
    base_y = hand_landmarks[0].y
    features = []
    for landmark in hand_landmarks:
        features.extend([landmark.x - base_x, landmark.y - base_y])
    return np.array(features, dtype=np.float32)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/recognition")
def recognition():
    return render_template("recognition.html")

@app.route("/how-it-works")
def how_it_works():
    return render_template("how_it_works.html")

@app.route("/analytics")
def analytics():
    return render_template("analytics.html")

@app.route("/about")
def about():
    return render_template("about.html")

@app.route("/contact")
def contact():
    return render_template("contact.html")

@app.route("/status")
def status():
    return jsonify({
        "mediapipe": landmarker is not None,
        "model_loaded": model is not None,
        "camera": True,  # Camera belongs to the user's browser.
    })

@app.route("/prediction", methods=["POST"])
def prediction():
    if landmarker is None:
        return jsonify({"label": "MediaPipe Error", "emoji": "⚠️", "landmarks": []}), 500
    if model is None:
        return jsonify({"label": "Model Error", "emoji": "⚠️", "landmarks": []}), 500

    try:
        data = request.get_json(silent=True) or {}
        image_data = data.get("image", "")

        if "," in image_data:
            image_data = image_data.split(",", 1)[1]

        if not image_data:
            return jsonify({"label": "No Hand", "emoji": "🤚", "landmarks": []})

        raw = base64.b64decode(image_data)
        frame = cv2.imdecode(np.frombuffer(raw, dtype=np.uint8), cv2.IMREAD_COLOR)

        if frame is None:
            return jsonify({"label": "No Hand", "emoji": "🤚", "landmarks": []})

        # Keep enough resolution for reliable hand detection on cloud deployment.
        # Browser frames may arrive compressed or smaller than expected.
        h, w = frame.shape[:2]
        if max(h, w) < 640:
            scale = 640 / max(h, w)
            frame = cv2.resize(frame, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_LINEAR)

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = landmarker.detect(mp_image)

        if not result.hand_landmarks:
            return jsonify({"label": "No Hand", "emoji": "🤚", "landmarks": []})

        hand = result.hand_landmarks[0]
        features = extract_landmarks(hand)
        label = str(model.predict([features])[0])
        landmarks = [{"x": lm.x, "y": lm.y} for lm in hand]

        return jsonify({
            "label": label,
            "emoji": GESTURE_EMOJIS.get(label, "🤚"),
            "landmarks": landmarks,
        })

    except Exception as e:
        print("Prediction error:", e)
        return jsonify({
            "label": "Detection Error",
            "emoji": "⚠️",
            "landmarks": []
        }), 500

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
