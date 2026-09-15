from flask import Flask, render_template, jsonify, request, send_from_directory
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import mediapipe as mp
import joblib
import numpy as np
import cv2
import os
import base64


# =========================================================
# PATHS
# =========================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

STATIC_DIR = os.path.join(BASE_DIR, "static")
TEMPLATE_DIR = os.path.join(BASE_DIR, "templates")

MODEL_PATH = os.path.join(
    BASE_DIR,
    "models",
    "gesture_model.pkl"
)

LANDMARKER_PATH = os.path.join(
    BASE_DIR,
    "models",
    "hand_landmarker.task"
)


# =========================================================
# FLASK APP
# =========================================================

app = Flask(
    __name__,
    static_folder=STATIC_DIR,
    template_folder=TEMPLATE_DIR
)


# =========================================================
# GLOBAL MODELS
# =========================================================

model = None
landmarker = None


# =========================================================
# GESTURE EMOJIS
# =========================================================

GESTURE_EMOJIS = {
    "Open_Palm": "🖐️",
    "Fist": "✊",
    "Thumbs_Up": "👍",
    "Thumbs_Down": "👎",
    "Peace": "✌️",
    "OK": "👌",

    # Pointing model label is displayed as 2 Fingers
    "Pointing": "✌️",

    "Love": "🤟",
    "Rock": "🤘",

    # One Finger is displayed as Pointing
    "One_Finger": "☝️",
    "One Finger": "☝️"
}


# =========================================================
# LOAD ML MODEL
# =========================================================

def load_model():

    global model

    try:

        if not os.path.exists(MODEL_PATH):

            print("ERROR: Gesture model not found:")
            print(MODEL_PATH)

            model = None
            return

        model = joblib.load(MODEL_PATH)

        print("======================================")
        print("Gesture model loaded successfully.")
        print("Model path:", MODEL_PATH)
        print("======================================")

    except Exception as e:

        model = None

        print("======================================")
        print("ERROR loading gesture model")
        print(str(e))
        print("======================================")


# =========================================================
# LOAD MEDIAPIPE HAND LANDMARKER
# =========================================================

def load_landmarker():

    global landmarker

    try:

        if not os.path.exists(LANDMARKER_PATH):

            print("ERROR: hand_landmarker.task not found:")
            print(LANDMARKER_PATH)

            landmarker = None
            return

        base_options = python.BaseOptions(
            model_asset_path=LANDMARKER_PATH
        )

        options = vision.HandLandmarkerOptions(

            base_options=base_options,

            running_mode=vision.RunningMode.IMAGE,

            num_hands=1,

            min_hand_detection_confidence=0.2,

            min_hand_presence_confidence=0.2,

            min_tracking_confidence=0.2
        )

        landmarker = vision.HandLandmarker.create_from_options(
            options
        )

        print("======================================")
        print("MediaPipe Hand Landmarker loaded.")
        print("Landmarker path:", LANDMARKER_PATH)
        print("======================================")

    except Exception as e:

        landmarker = None

        print("======================================")
        print("ERROR loading MediaPipe")
        print(str(e))
        print("======================================")


# =========================================================
# LOAD EVERYTHING ONCE
# =========================================================

load_model()
load_landmarker()


# =========================================================
# STATIC FILES
# =========================================================

@app.route("/static/<path:filename>")
def static_files(filename):

    return send_from_directory(
        STATIC_DIR,
        filename
    )


# =========================================================
# LANDMARK FEATURE EXTRACTION
# =========================================================

def extract_landmarks(hand_landmarks):

    if not hand_landmarks:

        return np.array([], dtype=np.float32)

    base_x = hand_landmarks[0].x
    base_y = hand_landmarks[0].y

    features = []

    for landmark in hand_landmarks:

        features.extend([
            landmark.x - base_x,
            landmark.y - base_y
        ])

    return np.array(
        features,
        dtype=np.float32
    )


# =========================================================
# HOME
# =========================================================

@app.route("/")
def home():

    return render_template(
        "index.html"
    )


# =========================================================
# RECOGNITION
# =========================================================

@app.route("/recognition")
def recognition():

    return render_template(
        "recognition.html"
    )


# =========================================================
# HOW IT WORKS
# =========================================================

@app.route("/how-it-works")
def how_it_works():

    return render_template(
        "how_it_works.html"
    )


# =========================================================
# ANALYTICS
# =========================================================

@app.route("/analytics")
def analytics():

    return render_template(
        "analytics.html"
    )


# =========================================================
# ABOUT
# =========================================================

@app.route("/about")
def about():

    return render_template(
        "about.html"
    )


# =========================================================
# CONTACT
# =========================================================

@app.route("/contact")
def contact():

    return render_template(
        "contact.html"
    )


# =========================================================
# STATUS CHECK
# =========================================================

@app.route("/status")
def status():

    return jsonify({

        "camera": True,

        "mediapipe": landmarker is not None,

        "model_loaded": model is not None

    })


# =========================================================
# PREDICTION
# =========================================================

@app.route("/prediction", methods=["POST"])
def prediction():

    # -----------------------------------------
    # Check MediaPipe
    # -----------------------------------------

    if landmarker is None:

        return jsonify({

            "label": "MediaPipe Error",

            "gesture": "MediaPipe Error",

            "emoji": "⚠️",

            "confidence": 0,

            "landmarks": []

        }), 500


    # -----------------------------------------
    # Check ML model
    # -----------------------------------------

    if model is None:

        return jsonify({

            "label": "Model Error",

            "gesture": "Model Error",

            "emoji": "⚠️",

            "confidence": 0,

            "landmarks": []

        }), 500


    try:

        # -----------------------------------------
        # Get JSON
        # -----------------------------------------

        data = request.get_json(
            silent=True
        ) or {}

        image_data = data.get(
            "image",
            ""
        )


        # -----------------------------------------
        # Remove Base64 prefix
        # -----------------------------------------

        if "," in image_data:

            image_data = image_data.split(
                ",",
                1
            )[1]


        # -----------------------------------------
        # Empty image
        # -----------------------------------------

        if not image_data:

            return jsonify({

                "label": "No Hand",

                "gesture": "No Hand",

                "emoji": "🤚",

                "confidence": 0,

                "landmarks": []

            })


        # -----------------------------------------
        # Decode image
        # -----------------------------------------

        try:

            raw = base64.b64decode(
                image_data,
                validate=True
            )

        except Exception:

            return jsonify({

                "label": "Invalid Image",

                "gesture": "Invalid Image",

                "emoji": "⚠️",

                "confidence": 0,

                "landmarks": []

            }), 400


        # -----------------------------------------
        # Convert image
        # -----------------------------------------

        frame = cv2.imdecode(

            np.frombuffer(
                raw,
                dtype=np.uint8
            ),

            cv2.IMREAD_COLOR
        )


        # -----------------------------------------
        # Image decoding failed
        # -----------------------------------------

        if frame is None:

            return jsonify({

                "label": "No Hand",

                "gesture": "No Hand",

                "emoji": "🤚",

                "confidence": 0,

                "landmarks": []

            })


        # -----------------------------------------
        # Keep image reasonably small
        # -----------------------------------------

        h, w = frame.shape[:2]

        max_dimension = max(
            h,
            w
        )

        if max_dimension > 480:

            scale = 480 / max_dimension

            new_width = int(
                w * scale
            )

            new_height = int(
                h * scale
            )

            frame = cv2.resize(

                frame,

                (
                    new_width,
                    new_height
                ),

                interpolation=cv2.INTER_AREA
            )


        # -----------------------------------------
        # BGR -> RGB
        # -----------------------------------------

        rgb = cv2.cvtColor(

            frame,

            cv2.COLOR_BGR2RGB
        )


        # -----------------------------------------
        # MediaPipe Image
        # -----------------------------------------

        mp_image = mp.Image(

            image_format=mp.ImageFormat.SRGB,

            data=rgb
        )


        # -----------------------------------------
        # Detect hand
        # -----------------------------------------

        result = landmarker.detect(
            mp_image
        )


        # -----------------------------------------
        # No hand detected
        # -----------------------------------------

        if not result.hand_landmarks:

            print(
                "MediaPipe: No hand detected"
            )

            return jsonify({

                "label": "No Hand",

                "gesture": "No Hand",

                "emoji": "🤚",

                "confidence": 0,

                "landmarks": []

            })


        # -----------------------------------------
        # First hand
        # -----------------------------------------

        hand = result.hand_landmarks[0]


        # -----------------------------------------
        # Extract features
        # -----------------------------------------

        features = extract_landmarks(
            hand
        )


        if features.size == 0:

            return jsonify({

                "label": "No Hand",

                "gesture": "No Hand",

                "emoji": "🤚",

                "confidence": 0,

                "landmarks": []

            })


        # -----------------------------------------
        # ML Prediction
        # -----------------------------------------

        label = str(
            model.predict(
                [features]
            )[0]
        )


        # -----------------------------------------
        # Confidence
        # -----------------------------------------

        confidence = 0.0

        if hasattr(
            model,
            "predict_proba"
        ):

            probabilities = model.predict_proba(
                [features]
            )[0]

            confidence = (
                float(
                    np.max(
                        probabilities
                    )
                ) * 100
            )


        # -----------------------------------------
        # Landmarks for frontend
        # -----------------------------------------

        landmarks = [

            {
                "x": float(
                    landmark.x
                ),

                "y": float(
                    landmark.y
                )
            }

            for landmark in hand

        ]


        # -----------------------------------------
        # Display label
        # -----------------------------------------

        display_label = label

        if label == "Pointing":

            display_label = "2 Fingers"

        elif label in [
            "One_Finger",
            "One Finger"
        ]:

            display_label = "Pointing"


        # -----------------------------------------
        # Emoji
        # -----------------------------------------

        emoji = GESTURE_EMOJIS.get(

            label,

            "🤚"
        )


        # -----------------------------------------
        # Final response
        # -----------------------------------------

        return jsonify({

            # Original model label
            "label": label,

            # Frontend-friendly display label
            "gesture": display_label,

            "emoji": emoji,

            "confidence": round(
                confidence,
                2
            ),

            "landmarks": landmarks

        })


    except Exception as e:

        print("======================================")
        print("Prediction error:")
        print(str(e))
        print("======================================")


        return jsonify({

            "label": "Detection Error",

            "gesture": "Detection Error",

            "emoji": "⚠️",

            "confidence": 0,

            "landmarks": []

        }), 500


# =========================================================
# RUN APP
# =========================================================

if __name__ == "__main__":

    port = int(
        os.environ.get(
            "PORT",
            5000
        )
    )

    app.run(

        host="0.0.0.0",

        port=port,

        debug=False
    )