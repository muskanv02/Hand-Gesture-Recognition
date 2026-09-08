import pandas as pd
import os
import joblib

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score


# ==============================
# FILE PATHS
# ==============================

CSV_FILE = "data/gesture_data.csv"
MODEL_DIR = "models"
MODEL_FILE = "models/gesture_model.pkl"


# ==============================
# CHECK CSV
# ==============================

if not os.path.exists(CSV_FILE):
    print("ERROR: gesture_data.csv not found!")
    exit()

print("Loading dataset...")


# ==============================
# LOAD DATA
# ==============================

df = pd.read_csv(CSV_FILE, header=None)

print("Dataset loaded!")
print("Total samples:", len(df))


# ==============================
# FEATURES & LABEL
# ==============================

X = df.iloc[:, :-1]
y = df.iloc[:, -1]


print("Number of features:", X.shape[1])
print("Number of gestures:", y.nunique())

print("\nGesture samples:")
print(y.value_counts())


# ==============================
# TRAIN / TEST SPLIT
# ==============================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


# ==============================
# MODEL
# ==============================

print("\nTraining model...")

model = RandomForestClassifier(
    n_estimators=200,
    random_state=42
)

model.fit(X_train, y_train)


# ==============================
# ACCURACY
# ==============================

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)

print("\n================================")
print("MODEL TRAINING COMPLETE")
print("================================")
print(f"Accuracy: {accuracy * 100:.2f}%")
print("================================")


# ==============================
# SAVE MODEL
# ==============================

os.makedirs(MODEL_DIR, exist_ok=True)

joblib.dump(model, MODEL_FILE)

print("\nModel saved successfully!")
print("Location:", MODEL_FILE)