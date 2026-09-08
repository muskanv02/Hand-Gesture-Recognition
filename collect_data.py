import cv2
import mediapipe as mp
import csv
import os

from mediapipe.tasks import python
from mediapipe.tasks.python import vision


# ==============================
# GESTURES
# ==============================

GESTURES = {
    "1": "Open_Palm",
    "2": "Fist",
    "3": "Thumbs_Up",
    "4": "Thumbs_Down",
    "5": "Peace",
    "6": "OK",
    "7": "Pointing",
    "8": "Love",
    "9": "Rock",
    "10": "One_Finger"
}


# ==============================
# MODEL PATH
# ==============================

MODEL_PATH = "models/hand_landmarker.task"

if not os.path.exists(MODEL_PATH):
    print("ERROR: hand_landmarker.task not found!")
    exit()


# ==============================
# CSV
# ==============================

os.makedirs("data", exist_ok=True)

CSV_FILE = "data/gesture_data.csv"


# ==============================
# MEDIAPIPE
# ==============================

base_options = python.BaseOptions(
    model_asset_path=MODEL_PATH
)

options = vision.HandLandmarkerOptions(
    base_options=base_options,
    num_hands=1,
    min_hand_detection_confidence=0.5,
    min_hand_presence_confidence=0.5,
    min_tracking_confidence=0.5
)

landmarker = vision.HandLandmarker.create_from_options(options)


# ==============================
# CAMERA
# ==============================

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("ERROR: Camera could not be opened.")
    landmarker.close()
    exit()


# ==============================
# LANDMARK CONNECTIONS
# ==============================

CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20),
    (0, 17)
]


# ==============================
# FUNCTIONS
# ==============================

def extract_landmarks(hand):
    """
    Convert 21 hand landmarks into
    normalized x,y features.
    """

    base_x = hand[0].x
    base_y = hand[0].y

    features = []

    for landmark in hand:
        features.append(landmark.x - base_x)
        features.append(landmark.y - base_y)

    return features


def draw_hand(frame, hand):

    h, w, _ = frame.shape

    points = []

    for landmark in hand:

        x = int(landmark.x * w)
        y = int(landmark.y * h)

        points.append((x, y))

        cv2.circle(
            frame,
            (x, y),
            5,
            (0, 255, 0),
            -1
        )

    for start, end in CONNECTIONS:

        cv2.line(
            frame,
            points[start],
            points[end],
            (255, 255, 255),
            2
        )


# ==============================
# START
# ==============================

print("\n====================================")
print("       HAND GESTURE DATA COLLECTION")
print("====================================\n")

print("Controls:")
print("1 - Open_Palm")
print("2 - Fist")
print("3 - Thumbs_Up")
print("4 - Thumbs_Down")
print("5 - Peace")
print("6 - OK")
print("7 - Pointing")
print("8 - Love")
print("9 - Rock")
print("0 - One_Finger")
print("Q - Quit\n")

print("Camera starting...")
print("Press a number to start collecting that gesture.\n")


# ==============================
# CURRENT GESTURE
# ==============================

current_gesture = None
count = 0


# ==============================
# OPEN CSV
# ==============================

with open(CSV_FILE, "a", newline="") as file:

    writer = csv.writer(file)

    while True:

        success, frame = cap.read()

        if not success:
            print("Camera frame could not be read.")
            break

        # Mirror camera
        frame = cv2.flip(frame, 1)

        # Convert BGR -> RGB
        rgb_frame = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB
        )

        # MediaPipe image
        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=rgb_frame
        )

        # Detect hand
        result = landmarker.detect(mp_image)


        # ==============================
        # HAND DETECTED
        # ==============================

        if result.hand_landmarks:

            hand = result.hand_landmarks[0]

            draw_hand(frame, hand)

            features = extract_landmarks(hand)

            # Only save if gesture selected
            if current_gesture is not None:

                writer.writerow(
                    features + [current_gesture]
                )

                file.flush()

                count += 1


        # ==============================
        # UI
        # ==============================

        cv2.rectangle(
            frame,
            (10, 10),
            (520, 135),
            (20, 20, 20),
            -1
        )


        if current_gesture is None:

            cv2.putText(
                frame,
                "Press 1-0 to select gesture",
                (25, 45),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (255, 255, 255),
                2
            )

        else:

            cv2.putText(
                frame,
                f"Gesture: {current_gesture}",
                (25, 45),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 255, 255),
                2
            )

            cv2.putText(
                frame,
                f"Samples: {count}",
                (25, 78),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2
            )

            cv2.putText(
                frame,
                "Press another number to change",
                (25, 105),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                1
            )


        cv2.imshow(
            "Gesture Data Collection",
            frame
        )


        # ==============================
        # KEYBOARD
        # ==============================

        key = cv2.waitKey(1) & 0xFF


        # Q = QUIT
        if key == ord("q"):

            print("\nCollection stopped.")
            break


        # ==============================
        # GESTURE SELECTION
        # ==============================

        if key == ord("1"):

            current_gesture = GESTURES["1"]
            count = 0
            print(f"\nStarted collecting: {current_gesture}")


        elif key == ord("2"):

            current_gesture = GESTURES["2"]
            count = 0
            print(f"\nStarted collecting: {current_gesture}")


        elif key == ord("3"):

            current_gesture = GESTURES["3"]
            count = 0
            print(f"\nStarted collecting: {current_gesture}")


        elif key == ord("4"):

            current_gesture = GESTURES["4"]
            count = 0
            print(f"\nStarted collecting: {current_gesture}")


        elif key == ord("5"):

            current_gesture = GESTURES["5"]
            count = 0
            print(f"\nStarted collecting: {current_gesture}")


        elif key == ord("6"):

            current_gesture = GESTURES["6"]
            count = 0
            print(f"\nStarted collecting: {current_gesture}")


        elif key == ord("7"):

            current_gesture = GESTURES["7"]
            count = 0
            print(f"\nStarted collecting: {current_gesture}")


        elif key == ord("8"):

            current_gesture = GESTURES["8"]
            count = 0
            print(f"\nStarted collecting: {current_gesture}")


        elif key == ord("9"):

            current_gesture = GESTURES["9"]
            count = 0
            print(f"\nStarted collecting: {current_gesture}")


        elif key == ord("0"):

            current_gesture = GESTURES["10"]
            count = 0
            print(f"\nStarted collecting: {current_gesture}")


# ==============================
# CLEANUP
# ==============================

cap.release()
cv2.destroyAllWindows()
landmarker.close()


print("\n====================================")
print("       DATA COLLECTION FINISHED")
print("====================================")
print(f"Data file: {CSV_FILE}")
print("All collected data has been saved.")
print("====================================")