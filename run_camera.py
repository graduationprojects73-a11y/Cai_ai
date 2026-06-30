import cv2
from ultralytics import YOLO
import time
import requests
import numpy as np
import threading

# إعدادات الـ API الخاصة بك على Hugging Face
HF_API_PROCESS = "https://graduationprojects73-violationn.hf.space/process"
HF_API_REMOVE  = "https://graduationprojects73-violationn.hf.space/remove_vehicle"

# تحميل موديل تتبع السيارات فقط المستعمل محلياً لتقليل الضغط
vehicle_model = YOLO("yolo11n.pt") 

# فتح الكاميرا
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
if not cap.isOpened():
    cap = cv2.VideoCapture(1, cv2.CAP_DSHOW)

processed_vehicle_ids = set()
vehicle_results_cache = {}

print(f"بدء النظام الذكي: التتبع محلي - الذكاء الاصطناعي على السحابة...")

last_ai_time = 0
ai_interval = 0.3

# دالة إرسال الصورة للسيرفر (تعمل في الخلفية لكي لا تبطئ الكاميرا)
def send_to_hf_api(track_id, vehicle_crop):
    try:
        _, img_encoded = cv2.imencode('.png', vehicle_crop)
        files = {'file': ('car.png', img_encoded.tobytes(), 'image/png')}
        data = {'track_id': str(track_id)}
        
        response = requests.post(HF_API_PROCESS, data=data, files=files, timeout=5)
        if response.status_code == 200:
            result = response.json()
            if result.get("status") == "Confirmed":
                plate = result.get("plate", "")
                processed_vehicle_ids.add(track_id)
                vehicle_results_cache[track_id] = plate
                print(f"[*] [مؤكد من السيرفر] -> ID: {track_id} | النمرة: {plate}")
    except Exception as e:
        pass # تجاهل أخطاء الشبكة المؤقتة

def notify_hf_remove(track_id):
    try:
        requests.post(HF_API_REMOVE, data={'track_id': str(track_id)}, timeout=2)
    except:
        pass

# لتتبع السيارات المختفية
currently_tracked_ids = set()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    current_time = time.time()
    display_frame = frame.copy()

    vehicle_tracks = vehicle_model.track(
        frame, 
        persist=True, 
        verbose=False, 
        classes=[2, 3, 5, 7],
        tracker="botsort.yaml" 
    )

    current_frame_ids = set()

    if (current_time - last_ai_time) >= ai_interval:
        last_ai_time = current_time
        
        if vehicle_tracks[0].boxes.id is not None:
            boxes = vehicle_tracks[0].boxes.xyxy.cpu().numpy()
            track_ids = vehicle_tracks[0].boxes.id.int().cpu().tolist()
            
            for box, track_id in zip(boxes, track_ids):
                current_frame_ids.add(track_id)
                
                # إرسال الصورة للسيرفر إذا لم يتم تأكيد النمرة بعد
                if track_id not in processed_vehicle_ids:
                    x1, y1, x2, y2 = map(int, box)
                    vehicle_crop = frame[max(0, int(y1)):min(frame.shape[0], int(y2)), 
                                         max(0, int(x1)):min(frame.shape[1], int(x2))]
                    
                    if vehicle_crop.size > 0:
                        alpha = 1.2
                        beta = 10
                        vehicle_crop = cv2.convertScaleAbs(vehicle_crop, alpha=alpha, beta=beta)
                        # تشغيل الإرسال في Thread منفصل لكي لا يقطع عرض الفيديو
                        threading.Thread(target=send_to_hf_api, args=(track_id, vehicle_crop)).start()

    # تنظيف السيارات التي اختفت من الشاشة
    disappeared_ids = currently_tracked_ids - current_frame_ids
    for track_id in disappeared_ids:
        if track_id not in processed_vehicle_ids: # إشعار السيرفر فقط لو كنا مازلنا نعالجها واختفت
            threading.Thread(target=notify_hf_remove, args=(track_id,)).start()
            
    currently_tracked_ids = current_frame_ids

    # الرسم على الشاشة 
    if vehicle_tracks[0].boxes.id is not None:
        boxes = vehicle_tracks[0].boxes.xyxy.cpu().numpy()
        track_ids = vehicle_tracks[0].boxes.id.int().cpu().tolist()
        
        for box, track_id in zip(boxes, track_ids):
            x1, y1, x2, y2 = map(int, box)
            color = (0, 255, 0) if track_id in processed_vehicle_ids else (0, 165, 255)
            
            cv2.rectangle(display_frame, (x1, y1), (x2, y2), color, 2)
            label = f"Car {track_id}"
            if track_id in vehicle_results_cache:
                label += f" | Plate: {vehicle_results_cache[track_id]}"
            cv2.putText(display_frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    cv2.imshow("Cloud Tracking System", display_frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
