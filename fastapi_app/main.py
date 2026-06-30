from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from ultralytics import YOLO
import io
import cv2
import requests
import numpy as np
import time
import re
import sqlite3
from collections import Counter

app = FastAPI(title="Egyptian Plate OCR API")

# تحميل الموديلات
plate_model = YOLO("best.pt")
char_model = YOLO("characters_yolo12n_best.pt")

# إعدادات الـ API الخارجي
EXTERNAL_API_URL = "http://72.62.186.142:8000/api/vehicle-log"
CAMERA_NUMBER = "1"
REQUIRED_VOTES = 3

DB_PATH = "vehicle_cache.db"

def init_db():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS readings (
            track_id INTEGER,
            plate_text TEXT,
            timestamp REAL
        )
    ''')
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS processed_vehicles (
            track_id INTEGER PRIMARY KEY,
            plate_text TEXT,
            timestamp REAL
        )
    ''')
    conn.commit()
    conn.close()

def add_reading(track_id: int, plate_text: str):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute(
        "INSERT INTO readings (track_id, plate_text, timestamp) VALUES (?, ?, ?)",
        (track_id, plate_text, time.time())
    )
    conn.commit()
    cursor.execute("SELECT plate_text FROM readings WHERE track_id = ?", (track_id,))
    readings = [row[0] for row in cursor.fetchall()]
    conn.close()
    return readings

def mark_as_processed(track_id: int, plate_text: str):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute(
        "INSERT OR REPLACE INTO processed_vehicles (track_id, plate_text, timestamp) VALUES (?, ?, ?)",
        (track_id, plate_text, time.time())
    )
    conn.commit()
    conn.close()

def is_processed(track_id: int) -> bool:
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT 1 FROM processed_vehicles WHERE track_id = ?", (track_id,))
    res = cursor.fetchone() is not None
    conn.close()
    return res

def remove_vehicle_from_cache(track_id: int):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("DELETE FROM readings WHERE track_id = ?", (track_id,))
    cursor.execute("DELETE FROM processed_vehicles WHERE track_id = ?", (track_id,))
    conn.commit()
    conn.close()

def clean_expired_cache(max_age_seconds=600):
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        expiry_time = time.time() - max_age_seconds
        cursor.execute("DELETE FROM readings WHERE timestamp < ?", (expiry_time,))
        cursor.execute("DELETE FROM processed_vehicles WHERE timestamp < ?", (expiry_time,))
        conn.commit()
        conn.close()
    except Exception as e:
        print(f"[Cache Cleanup Error] {e}")

def is_valid_egyptian_plate(text: str) -> bool:
    if not (5 <= len(text) <= 7):
        return False
        
    allowed_chars_pattern = re.compile(r'^[أببتثجحخدذرزسشصضطظعغفقكلمنھوي0-9]+$')
    if not allowed_chars_pattern.match(text):
        return False
        
    letters = re.findall(r'[أببتثجحخدذرزسشصضطظعغفقكلمنھوي]', text)
    digits = re.findall(r'[0-9]', text)
    
    if len(letters) in [2, 3] and len(digits) in [3, 4]:
        return True
        
    return False

def deskew_plate(plate_img):
    try:
        gray = cv2.cvtColor(plate_img, cv2.COLOR_BGR2GRAY)
        blurred = cv2.bilateralFilter(gray, 9, 75, 75)
        _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        coords = np.column_stack(np.where(thresh > 0))
        if len(coords) == 0:
            return plate_img
            
        rect = cv2.minAreaRect(coords)
        angle = rect[-1]
        
        if angle < -45:
            angle = -(90 + angle)
        else:
            angle = -angle
            
        if 1.0 < abs(angle) < 30.0:
            (h, w) = plate_img.shape[:2]
            center = (w // 2, h // 2)
            M = cv2.getRotationMatrix2D(center, angle, 1.0)
            rotated = cv2.warpAffine(plate_img, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
            return rotated
    except Exception as e:
        print(f"[Deskew Error] {e}")
    return plate_img

@app.on_event("startup")
def startup_event():
    init_db()
    # تنظيف أي بيانات قديمة لبدء تشغيل نظيف
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute("DELETE FROM readings")
        cursor.execute("DELETE FROM processed_vehicles")
        conn.commit()
        conn.close()
    except Exception as e:
        print(f"[Startup Clean Error] {e}")

def send_to_external_api(plate_text, image_np):
    try:
        _, img_encoded = cv2.imencode('.png', image_np)
        files = {'image': ('plate.png', img_encoded.tobytes(), 'image/png')}
        data = {'license_plate': plate_text, 'number_camera': CAMERA_NUMBER}
        response = requests.post(EXTERNAL_API_URL, data=data, files=files, timeout=5)
        print(f"[API] إرسال إلى قاعدة البيانات: {plate_text} - الحالة: {response.status_code}")
    except Exception as e:
        print(f"[API] خطأ في الإرسال: {e}")

@app.get("/")
def read_root():
    return {"status": "Online", "mode": "Server-Side Voting"}

@app.post("/remove_vehicle")
async def remove_vehicle(track_id: int = Form(...)):
    """ Endpoint لتنظيف ذاكرة السيرفر عندما تختفي السيارة من الكاميرا """
    remove_vehicle_from_cache(track_id)
    return {"status": "removed"}

@app.post("/process")
async def process_image(
    file: UploadFile = File(...),
    track_id: int = Form(...) # الكاميرا يجب أن ترسل الـ ID مع الصورة
):
    # تنظيف دوري للذاكرة العشوائية لتجنب التضخم
    clean_expired_cache()

    # إذا تم تأكيد هذه السيارة مسبقاً، تجاهل الصور الجديدة لتخفيف الضغط
    if is_processed(track_id):
        return {"status": "Already Processed"}

    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    if img is None:
        raise HTTPException(status_code=400, detail="Invalid image file")

    plate_detections = plate_model(img, verbose=False)

    for p_res in plate_detections:
        for p_box in p_res.boxes:
            bx1, by1, bx2, by2 = map(int, p_box.xyxy[0])
            plate_crop = img[by1:by2, bx1:bx2]
            
            if plate_crop.size > 0:
                # تصحيح ميل اللوحة
                plate_crop = deskew_plate(plate_crop)
                
                char_res = char_model(plate_crop, verbose=False)
                detected_chars = []
                for c_res in char_res:
                    for c_box in c_res.boxes:
                        detected_chars.append((float(c_box.xyxy[0][0]), char_model.names[int(c_box.cls[0])]))
                
                detected_chars.sort(key=lambda x: x[0])
                plate_text = "".join([c[1] for c in detected_chars])

                # التحقق من صيغة اللوحة المصرية
                if plate_text and is_valid_egyptian_plate(plate_text):
                    # نظام التصويت على السيرفر باستخدام SQLite
                    readings = add_reading(track_id, plate_text)
                    
                    if len(readings) >= REQUIRED_VOTES:
                        most_common_plate, count = Counter(readings).most_common(1)[0]
                        longest_plate = max(readings, key=len)
                        final_decision = most_common_plate if count >= 2 else longest_plate
                        
                        if len(final_decision) >= 4:
                            mark_as_processed(track_id, final_decision)
                            # نرسل الصورة كاملة (img) إلى الـ API الخارجي
                            send_to_external_api(final_decision, img)
                            return {"status": "Confirmed", "plate": final_decision}

    # الحصول على عدد المحاولات الحالي
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT COUNT(*) FROM readings WHERE track_id = ?", (track_id,))
    count = cursor.fetchone()[0]
    conn.close()

    return {"status": "Processing", "readings_count": count}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7860)
