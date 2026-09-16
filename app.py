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


# =========================================================
# PATHS
# =========================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

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
# GLOBAL VARIABLES
# =========================================================

model = None
landmarker = None


# =========================================================
# GESTURE EMOJIS
# =========================================================

GESTURE_EMOJIS = {

    "Open_Palm": "🖐️",
    "Open Palm": "🖐️",

    "Fist": "✊",

    "Thumbs_Up": "👍",
    "Thumbs Up": "👍",

    "Thumbs_Down": "👎",
    "Thumbs Down": "👎",

    "Peace": "✌️",

    "OK": "👌",

    # Model's Pointing class should display
    # as 2 Fingers according to your UI requirement
    "Pointing": "✌️",

    "Love": "🤟",

    "Rock": "🤘",

    "One_Finger": "☝️",
    "One Finger": "☝️"
}


# =========================================================
# LOAD MODEL + MEDIAPIPE
# =========================================================

def load_models():

    global model
    global landmarker

    # -------------------------
    # Load ML Model
    # -------------------------

    try:

        if not os.path.exists(MODEL_PATH):

            print(
                "MODEL FILE NOT FOUND:",
                MODEL_PATH
            )

        else:

            model = joblib.load(
                MODEL_PATH
            )

            print(
                "Gesture model loaded successfully."
            )

    except Exception as e:

        print(
            "MODEL LOAD ERROR:",
            repr(e)
        )

        model = None


    # -------------------------
    # Load MediaPipe
    # -------------------------

    try:

        if not os.path.exists(LANDMARKER_PATH):

            print(
                "HAND LANDMARKER FILE NOT FOUND:",
                LANDMARKER_PATH
            )

        else:

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


            landmarker = (
                vision.HandLandmarker
                .create_from_options(options)
            )


            print(
                "MediaPipe Hand Landmarker loaded successfully."
            )

    except Exception as e:

        print(
            "MEDIAPIPE LOAD ERROR:",
            repr(e)
        )

        landmarker = None


# Load everything when server starts
load_models()


# =========================================================
# FEATURE EXTRACTION
# =========================================================

def extract_landmarks(hand_landmarks):

    if not hand_landmarks:

        return np.zeros(
            42,
            dtype=np.float32
        )


    # Wrist = landmark 0
    wrist_x = hand_landmarks[0].x
    wrist_y = hand_landmarks[0].y


    features = []


    for landmark in hand_landmarks:

        relative_x = (
            landmark.x - wrist_x
        )

        relative_y = (
            landmark.y - wrist_y
        )

        features.extend([
            relative_x,
            relative_y
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
# AI TECHNOLOGY
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
# SYSTEM STATUS
# =========================================================

@app.route("/status")
def status():

    return jsonify({

        # Browser camera is handled by JavaScript
        "camera": True,

        # Backend MediaPipe status
        "mediapipe": landmarker is not None,

        # ML model status
        "model_loaded": model is not None
    })


# =========================================================
# PREDICTION API
# =========================================================

@app.route(
    "/prediction",
    methods=["POST"]
)
def prediction():

    # -------------------------
    # Check MediaPipe
    # -------------------------

    if landmarker is None:

        return jsonify({

            "success": False,

            "label": "MediaPipe Error",

            "emoji": "⚠️",

            "confidence": 0,

            "landmarks": [],

            "error": "MediaPipe is not available."
        }), 500


    # -------------------------
    # Check ML model
    # -------------------------

    if model is None:

        return jsonify({

            "success": False,

            "label": "Model Error",

            "emoji": "⚠️",

            "confidence": 0,

            "landmarks": [],

            "error": "Gesture model is not loaded."
        }), 500


    try:

        # =================================================
        # GET JSON
        # =================================================

        data = request.get_json(
            silent=True
        ) or {}


        image_data = data.get(
            "image",
            ""
        )


        if not image_data:

            return jsonify({

                "success": True,

                "label": "No Hand",

                "emoji": "🤚",

                "confidence": 0,

                "landmarks": []
            })


        # =================================================
        # REMOVE DATA URL PREFIX
        # =================================================

        if "," in image_data:

            image_data = image_data.split(
                ",",
                1
            )[1]


        # =================================================
        # DECODE IMAGE
        # =================================================

        try:

            image_bytes = base64.b64decode(
                image_data
            )

        except Exception:

            return jsonify({

                "success": False,

                "label": "Detection Error",

                "emoji": "⚠️",

                "confidence": 0,

                "landmarks": [],

                "error": "Invalid image data."
            }), 400


        # =================================================
        # CONVERT IMAGE TO OPENCV
        # =================================================

        np_array = np.frombuffer(
            image_bytes,
            dtype=np.uint8
        )


        frame = cv2.imdecode(
            np_array,
            cv2.IMREAD_COLOR
        )


        if frame is None:

            return jsonify({

                "success": True,

                "label": "No Hand",

                "emoji": "🤚",

                "confidence": 0,

                "landmarks": []
            })


        # =================================================
        # RESIZE FRAME
        # =================================================

        height, width = frame.shape[:2]


        max_dimension = 640


        if max(
            height,
            width
        ) > max_dimension:

            scale = (
                max_dimension /
                max(height, width)
            )


            new_width = max(
                1,
                int(width * scale)
            )


            new_height = max(
                1,
                int(height * scale)
            )


            frame = cv2.resize(

                frame,

                (
                    new_width,
                    new_height
                ),

                interpolation=cv2.INTER_AREA
            )


        # =================================================
        # BGR → RGB
        # =================================================

        rgb_frame = cv2.cvtColor(

            frame,

            cv2.COLOR_BGR2RGB
        )


        # =================================================
        # MEDIAPIPE IMAGE
        # =================================================

        mp_image = mp.Image(

            image_format=mp.ImageFormat.SRGB,

            data=rgb_frame
        )


        # =================================================
        # HAND DETECTION
        # =================================================

        result = landmarker.detect(
            mp_image
        )


        # =================================================
        # NO HAND
        # =================================================

        if not result.hand_landmarks:

            return jsonify({

                "success": True,

                "label": "No Hand",

                "emoji": "🤚",

                "confidence": 0,

                "landmarks": []
            })


        # =================================================
        # FIRST HAND
        # =================================================

        hand = result.hand_landmarks[0]


        # =================================================
        # EXTRACT FEATURES
        # =================================================

        features = extract_landmarks(
            hand
        )


        # =================================================
        # MODEL PREDICTION
        # =================================================

        prediction_result = model.predict(
            features.reshape(1, -1)
        )


        raw_label = str(
            prediction_result[0]
        )


        # =================================================
        # CONFIDENCE
        # =================================================

        confidence = 0.0


        if hasattr(
            model,
            "predict_proba"
        ):

            try:

                probabilities = (
                    model.predict_proba(
                        features.reshape(1, -1)
                    )[0]
                )


                confidence = float(
                    np.max(probabilities)
                )


            except Exception as confidence_error:

                print(
                    "CONFIDENCE ERROR:",
                    repr(confidence_error)
                )

                confidence = 0.0


        # Convert 0-1 → 0-100
        confidence_percent = (
            confidence * 100
        )


        # =================================================
        # LANDMARKS FOR FRONTEND
        # =================================================

        landmarks = []


        for landmark in hand:

            landmarks.append({

                "x": float(
                    landmark.x
                ),

                "y": float(
                    landmark.y
                )
            })


        # =================================================
        # EMOJI
        # =================================================

        emoji = GESTURE_EMOJIS.get(
            raw_label,
            "🤚"
        )


        # =================================================
        # FINAL RESPONSE
        # =================================================

        print(
            "GESTURE:",
            raw_label,
            "| CONFIDENCE:",
            round(
                confidence_percent,
                2
            )
        )


        return jsonify({

            "success": True,

            "label": raw_label,

            "gesture": raw_label,

            "emoji": emoji,

            "confidence": confidence_percent,

            "landmarks": landmarks
        })


    except Exception as e:

        print(
            "PREDICTION ERROR:",
            repr(e)
        )


        return jsonify({

            "success": False,

            "label": "Detection Error",

            "gesture": "Detection Error",

            "emoji": "⚠️",

            "confidence": 0,

            "landmarks": [],

            "error": str(e)
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

        debug=False,

        threaded=True
    )