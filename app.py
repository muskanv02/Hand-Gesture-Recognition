from flask import Flask, render_template, Response, jsonify
import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import joblib
import numpy as np
import os


# =========================================================
# FLASK APP
# =========================================================

app = Flask(__name__)


# =========================================================
# FILE PATHS
# =========================================================

MODEL_PATH = os.path.join(
    "models",
    "gesture_model.pkl"
)

LANDMARKER_PATH = os.path.join(
    "models",
    "hand_landmarker.task"
)


# =========================================================
# LOAD MACHINE LEARNING MODEL
# =========================================================

model = None

if os.path.exists(MODEL_PATH):

    try:

        model = joblib.load(MODEL_PATH)

        print("Gesture model loaded successfully.")

    except Exception as e:

        print("Error loading gesture model:", e)

else:

    print("Gesture model not found:", MODEL_PATH)


# =========================================================
# MEDIAPIPE HAND LANDMARKER
# =========================================================

landmarker = None

if os.path.exists(LANDMARKER_PATH):

    try:

        base_options = python.BaseOptions(
            model_asset_path=LANDMARKER_PATH
        )

        options = vision.HandLandmarkerOptions(

            base_options=base_options,

            num_hands=1,

            min_hand_detection_confidence=0.5,

            min_hand_presence_confidence=0.5,

            min_tracking_confidence=0.5

        )

        landmarker = vision.HandLandmarker.create_from_options(
            options
        )

        print("MediaPipe Hand Landmarker loaded successfully.")

    except Exception as e:

        print("Error loading MediaPipe:", e)

else:

    print(
        "hand_landmarker.task not found:",
        LANDMARKER_PATH
    )


# =========================================================
# CAMERA
# =========================================================

camera = cv2.VideoCapture(0)


# Camera resolution

camera.set(
    cv2.CAP_PROP_FRAME_WIDTH,
    640
)

camera.set(
    cv2.CAP_PROP_FRAME_HEIGHT,
    480
)


# =========================================================
# CURRENT GESTURE
# =========================================================

current_label = "No Hand"

current_emoji = "🤚"


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

    "Pointing": "☝️",

    "Love": "🤟",

    "Rock": "🤘",

    "One_Finger": "☝️"

}


# =========================================================
# MEDIAPIPE HAND CONNECTIONS
# =========================================================

CONNECTIONS = [

    (0, 1),
    (1, 2),
    (2, 3),
    (3, 4),

    (0, 5),
    (5, 6),
    (6, 7),
    (7, 8),

    (5, 9),
    (9, 10),
    (10, 11),
    (11, 12),

    (9, 13),
    (13, 14),
    (14, 15),
    (15, 16),

    (13, 17),
    (17, 18),
    (18, 19),
    (19, 20),

    (0, 17)

]


# =========================================================
# EXTRACT LANDMARK FEATURES
# =========================================================

def extract_landmarks(hand_landmarks):

    """
    Convert 21 hand landmarks into
    42 normalized x/y features.

    Landmark 0 (wrist) is used as
    the reference point.
    """

    base_x = hand_landmarks[0].x

    base_y = hand_landmarks[0].y


    features = []


    for landmark in hand_landmarks:

        features.append(
            landmark.x - base_x
        )

        features.append(
            landmark.y - base_y
        )


    return np.array(
        features,
        dtype=np.float32
    )


# =========================================================
# DRAW HAND LANDMARKS
# =========================================================

def draw_hand(frame, hand_landmarks):

    """
    Draw hand landmarks and connections
    on the OpenCV camera frame.
    """

    height, width, _ = frame.shape


    # -----------------------------------------------------
    # Draw connections
    # -----------------------------------------------------

    for start, end in CONNECTIONS:

        x1 = int(
            hand_landmarks[start].x * width
        )

        y1 = int(
            hand_landmarks[start].y * height
        )

        x2 = int(
            hand_landmarks[end].x * width
        )

        y2 = int(
            hand_landmarks[end].y * height
        )


        cv2.line(

            frame,

            (x1, y1),

            (x2, y2),

            (120, 90, 255),

            2

        )


    # -----------------------------------------------------
    # Draw landmark points
    # -----------------------------------------------------

    for landmark in hand_landmarks:

        x = int(
            landmark.x * width
        )

        y = int(
            landmark.y * height
        )


        cv2.circle(

            frame,

            (x, y),

            5,

            (255, 255, 255),

            -1

        )


# =========================================================
# GENERATE CAMERA FRAMES
# =========================================================

def generate_frames():

    global current_label
    global current_emoji


    while True:


        # -------------------------------------------------
        # Read camera
        # -------------------------------------------------

        success, frame = camera.read()


        if not success:

            print("Unable to read camera frame.")

            break


        # -------------------------------------------------
        # Mirror camera
        # -------------------------------------------------

        frame = cv2.flip(
            frame,
            1
        )


        # -------------------------------------------------
        # Default gesture
        # -------------------------------------------------

        label = "No Hand"


        # -------------------------------------------------
        # Check MediaPipe
        # -------------------------------------------------

        if landmarker is not None:


            try:

                # Convert BGR → RGB

                rgb_frame = cv2.cvtColor(

                    frame,

                    cv2.COLOR_BGR2RGB

                )


                # Create MediaPipe image

                mp_image = mp.Image(

                    image_format=mp.ImageFormat.SRGB,

                    data=rgb_frame

                )


                # Detect hand

                result = landmarker.detect(
                    mp_image
                )


                # -------------------------------------------------
                # If hand detected
                # -------------------------------------------------

                if result.hand_landmarks:

                    hand = result.hand_landmarks[0]


                    # Draw hand

                    draw_hand(

                        frame,

                        hand

                    )


                    # Extract features

                    features = extract_landmarks(
                        hand
                    )


                    # -------------------------------------------------
                    # Predict gesture
                    # -------------------------------------------------

                    if model is not None:

                        try:

                            prediction = model.predict(
                                [features]
                            )

                            label = str(
                                prediction[0]
                            )

                        except Exception as e:

                            print(
                                "Prediction Error:",
                                e
                            )

                            label = "Detection Error"


            except Exception as e:

                print(
                    "MediaPipe Detection Error:",
                    e
                )

                label = "Detection Error"


        else:

            label = "MediaPipe Error"


        # =====================================================
        # UPDATE CURRENT GESTURE
        # =====================================================

        current_label = label


        current_emoji = GESTURE_EMOJIS.get(

            label,

            "🤚"

        )


        # =====================================================
        # CAMERA OVERLAY
        # =====================================================

        cv2.rectangle(

            frame,

            (15, 15),

            (390, 82),

            (15, 15, 25),

            -1

        )


        cv2.putText(

            frame,

            f"Gesture: {label}",

            (28, 57),

            cv2.FONT_HERSHEY_SIMPLEX,

            0.82,

            (255, 255, 255),

            2,

            cv2.LINE_AA

        )


        # =====================================================
        # MEDIAPIPE STATUS
        # =====================================================

        status_text = "MediaPipe: Ready"


        if landmarker is None:

            status_text = "MediaPipe: Error"


        cv2.putText(

            frame,

            status_text,

            (28, frame.shape[0] - 20),

            cv2.FONT_HERSHEY_SIMPLEX,

            0.55,

            (220, 220, 220),

            1,

            cv2.LINE_AA

        )


        # =====================================================
        # ENCODE FRAME
        # =====================================================

        ret, buffer = cv2.imencode(

            ".jpg",

            frame

        )


        if not ret:

            continue


        frame_bytes = buffer.tobytes()


        # =====================================================
        # SEND FRAME TO BROWSER
        # =====================================================

        yield (

            b"--frame\r\n"

            b"Content-Type: image/jpeg\r\n\r\n"

            + frame_bytes

            + b"\r\n"

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
# LIVE RECOGNITION
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
# VIDEO FEED
# =========================================================

@app.route("/video_feed")
def video_feed():

    return Response(

        generate_frames(),

        mimetype=
        "multipart/x-mixed-replace; boundary=frame"

    )


# =========================================================
# SYSTEM STATUS
# =========================================================

@app.route("/status")
def status():

    return jsonify({

        "camera": camera.isOpened(),

        "mediapipe": landmarker is not None,

        "model_loaded": model is not None

    })


# =========================================================
# LIVE GESTURE PREDICTION
# =========================================================

@app.route("/prediction")
def prediction():

    return jsonify({

        "label": current_label,

        "emoji": current_emoji

    })


# =========================================================
# RUN APPLICATION
# =========================================================

if __name__ == "__main__":

    print()
    print("======================================")
    print("      GESTURE AI APPLICATION")
    print("======================================")
    print()

    print(
        "Camera:",
        "Ready" if camera.isOpened()
        else "Not Available"
    )

    print(
        "MediaPipe:",
        "Ready" if landmarker is not None
        else "Not Available"
    )

    print(
        "Gesture Model:",
        "Loaded" if model is not None
        else "Not Loaded"
    )

    print()
    print(
        "Open: http://127.0.0.1:5000"
    )
    print()

    app.run(

        debug=False,

        threaded=True

    )