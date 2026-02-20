import os
import pickle
from datetime import datetime, date
import pandas as pd
import numpy as np
import cv2

ENCODINGS_PKL = os.path.join(os.getcwd(), "encodings.pkl")
STUDENTS_CSV = os.path.join(os.getcwd(), "students.csv")
CHECKIN_CSV = os.path.join(os.getcwd(), "attendance_checkin.csv")
CHECKOUT_CSV = os.path.join(os.getcwd(), "attendance_checkout.csv")

def ensure_files():
    if not os.path.exists(ENCODINGS_PKL):
        with open(ENCODINGS_PKL, "wb") as f:
            pickle.dump([], f)
    if not os.path.exists(STUDENTS_CSV):
        pd.DataFrame(columns=["name", "roll_no", "email"]).to_csv(STUDENTS_CSV, index=False)
    if not os.path.exists(CHECKIN_CSV):
        pd.DataFrame(columns=["name", "roll_no", "email", "date", "check_in_time"]).to_csv(CHECKIN_CSV, index=False)
    if not os.path.exists(CHECKOUT_CSV):
        pd.DataFrame(columns=["name", "roll_no", "email", "date", "check_out_time"]).to_csv(CHECKOUT_CSV, index=False)

def load_known():
    ensure_files()
    with open(ENCODINGS_PKL, "rb") as f:
        items = pickle.load(f)
    encodings = [np.array(it["encoding"]) for it in items]
    meta = [{"name": it["name"], "roll_no": it["roll_no"], "email": it["email"]} for it in items]
    return encodings, meta

def match_encoding(encoding, known_encodings, threshold=0.3):
    if not known_encodings:
        return None, None
    query = np.array(encoding, dtype=np.float32)
    best_idx = None
    best_dist = 1.0
    for i, known in enumerate(known_encodings):
        k = np.array(known, dtype=np.float32)
        # Bhattacharyya distance between normalized histograms
        dist = cv2.compareHist(query, k, cv2.HISTCMP_BHATTACHARYYA)
        if dist < best_dist:
            best_dist = dist
            best_idx = i
    if best_idx is not None and best_dist <= threshold:
        return best_idx, best_dist
    return None, None

def bytes_to_bgr(image_bytes):
    arr = np.frombuffer(image_bytes, dtype=np.uint8)
    img_bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img_bgr is None:
        return None
    return img_bgr

def encode_single_face(image_bytes):
    img_bgr = bytes_to_bgr(image_bytes)
    if img_bgr is None:
        return None
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    clf = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    faces = clf.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60))
    if len(faces) == 0:
        return None
    x, y, w, h = max(faces, key=lambda r: r[2] * r[3])
    roi = gray[y:y+h, x:x+w]
    roi = cv2.resize(roi, (200, 200))
    hist = cv2.calcHist([roi], [0], None, [256], [0, 256]).flatten()
    hist = hist / (np.sum(hist) + 1e-8)
    return hist.tolist()

def already_marked(csv_path, roll_no, d):
    if not os.path.exists(csv_path):
        return False
    df = pd.read_csv(csv_path)
    if df.empty:
        return False
    return any((df["roll_no"].astype(str) == str(roll_no)) & (df["date"] == d))

def write_attendance(csv_path, name, roll_no, email, d, t, column_time):
    df = pd.read_csv(csv_path) if os.path.exists(csv_path) else pd.DataFrame(columns=["name", "roll_no", "email", "date", column_time])
    df.loc[len(df)] = [name, str(roll_no), email, d, t]
    df.to_csv(csv_path, index=False)

def mark_checkin(image_bytes, threshold=0.6):
    ensure_files()
    encodings, meta = load_known()
    encoding = encode_single_face(image_bytes)
    if encoding is None:
        return False, "Face Not Detected"
    idx, dist = match_encoding(encoding, encodings, threshold)
    if idx is None:
        return False, "Face Not Recognized"
    info = meta[idx]
    today = date.today().strftime("%Y-%m-%d")
    now = datetime.now().strftime("%H:%M:%S")
    if already_marked(CHECKIN_CSV, info["roll_no"], today):
        return True, f"Already checked in: {info['name']} ({info['roll_no']})"
    write_attendance(CHECKIN_CSV, info["name"], info["roll_no"], info["email"], today, now, "check_in_time")
    return True, f"Check-In marked: {info['name']} ({info['roll_no']}) at {now}"

def mark_checkout(image_bytes, threshold=0.6):
    ensure_files()
    encodings, meta = load_known()
    encoding = encode_single_face(image_bytes)
    if encoding is None:
        return False, "Face Not Detected"
    idx, dist = match_encoding(encoding, encodings, threshold)
    if idx is None:
        return False, "Face Not Recognized"
    info = meta[idx]
    today = date.today().strftime("%Y-%m-%d")
    now = datetime.now().strftime("%H:%M:%S")
    if already_marked(CHECKOUT_CSV, info["roll_no"], today):
        return True, f"Already checked out: {info['name']} ({info['roll_no']})"
    write_attendance(CHECKOUT_CSV, info["name"], info["roll_no"], info["email"], today, now, "check_out_time")
    return True, f"Check-Out marked: {info['name']} ({info['roll_no']}) at {now}"
