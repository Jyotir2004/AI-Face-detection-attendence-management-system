import os
import pickle
from datetime import datetime
import pandas as pd
import numpy as np
import cv2

DATA_DIR = os.path.join(os.getcwd(), "dataset")
STUDENTS_CSV = os.path.join(os.getcwd(), "students.csv")
ENCODINGS_PKL = os.path.join(os.getcwd(), "encodings.pkl")

def ensure_paths():
    os.makedirs(DATA_DIR, exist_ok=True)
    if not os.path.exists(STUDENTS_CSV):
        pd.DataFrame(columns=["name", "roll_no", "email"]).to_csv(STUDENTS_CSV, index=False)
    if not os.path.exists(ENCODINGS_PKL):
        with open(ENCODINGS_PKL, "wb") as f:
            pickle.dump([], f)

def load_encodings():
    ensure_paths()
    with open(ENCODINGS_PKL, "rb") as f:
        return pickle.load(f)

def save_encodings(items):
    with open(ENCODINGS_PKL, "wb") as f:
        pickle.dump(items, f)

def add_student_row(name, roll_no, email):
    df = pd.read_csv(STUDENTS_CSV) if os.path.exists(STUDENTS_CSV) else pd.DataFrame(columns=["name", "roll_no", "email"])
    if str(roll_no) in set(df["roll_no"].astype(str)):
        df.loc[df["roll_no"].astype(str) == str(roll_no), ["name", "email"]] = [name, email]
    else:
        df.loc[len(df)] = [name, str(roll_no), email]
    df.to_csv(STUDENTS_CSV, index=False)

def _detect_face_roi(img_bgr):
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    clf = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    faces = clf.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60))
    if len(faces) == 0:
        return None
    x, y, w, h = max(faces, key=lambda r: r[2] * r[3])
    roi = gray[y:y+h, x:x+w]
    roi = cv2.resize(roi, (200, 200))
    return roi

def _compute_encoding(roi_gray):
    hist = cv2.calcHist([roi_gray], [0], None, [256], [0, 256]).flatten()
    hist = hist / (np.sum(hist) + 1e-8)
    return hist.tolist()

def encode_image_bytes(image_bytes):
    arr = np.frombuffer(image_bytes, dtype=np.uint8)
    img_bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img_bgr is None:
        return None
    roi = _detect_face_roi(img_bgr)
    if roi is None:
        return None
    return _compute_encoding(roi)

def register_student(name, roll_no, email, images_bytes_list):
    ensure_paths()
    enc_list = load_encodings()
    saved_any = False
    for i, b in enumerate(images_bytes_list):
        encoding = encode_image_bytes(b)
        if encoding is None:
            continue
        ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        img_path = os.path.join(DATA_DIR, f"{roll_no}_{ts}_{i}.jpg")
        with open(img_path, "wb") as f:
            f.write(b)
        enc_list.append({"roll_no": str(roll_no), "name": name, "email": email, "encoding": encoding})
        saved_any = True
    if saved_any:
        save_encodings(enc_list)
        add_student_row(name, roll_no, email)
    return saved_any
