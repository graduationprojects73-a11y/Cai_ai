import cv2
import numpy as np
import re
import time
from ultralytics import YOLO

# ── تحميل الموديلات ─────────────────────────────────────────────────────────
print("[*] جاري تحميل الموديلات...")
vehicle_model = YOLO("yolo11n.pt")
plate_model   = YOLO("fastapi_app/best.pt")
char_model    = YOLO("fastapi_app/characters_yolo12n_best.pt")
print("[✓] تم تحميل الموديلات بنجاح\n")


def is_valid_egyptian_plate(text: str) -> bool:
    if not (5 <= len(text) <= 7):
        return False
    allowed = re.compile(r'^[أببتثجحخدذرزسشصضطظعغفقكلمنھوي0-9]+$')
    if not allowed.match(text):
        return False
    letters = re.findall(r'[أببتثجحخدذرزسشصضطظعغفقكلمنھوي]', text)
    digits  = re.findall(r'[0-9]', text)
    return len(letters) in [2, 3] and len(digits) in [3, 4]


def deskew_plate(plate_img):
    try:
        gray      = cv2.cvtColor(plate_img, cv2.COLOR_BGR2GRAY)
        blurred   = cv2.bilateralFilter(gray, 9, 75, 75)
        _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        coords    = np.column_stack(np.where(thresh > 0))
        if len(coords) == 0:
            return plate_img
        rect  = cv2.minAreaRect(coords)
        angle = rect[-1]
        if angle < -45:
            angle = -(90 + angle)
        else:
            angle = -angle
        if 1.0 < abs(angle) < 30.0:
            h, w = plate_img.shape[:2]
            M    = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1.0)
            return cv2.warpAffine(plate_img, M, (w, h),
                                  flags=cv2.INTER_CUBIC,
                                  borderMode=cv2.BORDER_REPLICATE)
    except Exception as e:
        print(f"[Deskew Error] {e}")
    return plate_img


def read_plate(vehicle_crop) -> str:
    """يقرأ النمرة من crop السيارة مباشرة بدون voting."""
    plate_results = plate_model(vehicle_crop, verbose=False)
    for p_res in plate_results:
        for p_box in p_res.boxes:
            bx1, by1, bx2, by2 = map(int, p_box.xyxy[0])
            plate_crop = vehicle_crop[by1:by2, bx1:bx2]
            if plate_crop.size == 0:
                continue

            plate_crop    = deskew_plate(plate_crop)
            char_res      = char_model(plate_crop, verbose=False)
            detected_chars = []
            for c_res in char_res:
                for c_box in c_res.boxes:
                    x_pos = float(c_box.xyxy[0][0])
                    label = char_model.names[int(c_box.cls[0])]
                    detected_chars.append((x_pos, label))

            detected_chars.sort(key=lambda x: x[0])
            plate_text = "".join(c[1] for c in detected_chars)

            if is_valid_egyptian_plate(plate_text):
                return plate_text
    return ""


def main():
    cap = None
    for idx in range(3):
        c = cv2.VideoCapture(idx, cv2.CAP_DSHOW)
        if c.isOpened():
            cap = c
            print(f"[Camera] فُتحت الكاميرا على index {idx}")
            break
    if cap is None:
        raise RuntimeError("فشل فتح الكاميرا — تأكد من توصيل كاميرا")

    # plate_cache: track_id -> آخر نمرة مقروءة
    plate_cache:  dict[int, str] = {}
    last_ai_time = 0.0
    AI_INTERVAL  = 0.5   # ثانية بين كل قراءة للوحة

    print("النظام شغّال — اضغط Q للإيقاف\n")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("[!] انقطع feed الكاميرا")
                break

            now     = time.time()
            display = frame.copy()

            tracks = vehicle_model.track(
                frame,
                persist=True,
                verbose=False,
                classes=[2, 3, 5, 7],
                tracker="bytetrack.yaml",
            )

            current_frame_ids: set[int] = set()
            boxes_list: list = []
            ids_list:   list = []

            if tracks[0].boxes.id is not None:
                raw_boxes = tracks[0].boxes.xyxy.cpu().numpy()
                raw_ids   = tracks[0].boxes.id.int().cpu().tolist()
                for box, tid in zip(raw_boxes, raw_ids):
                    current_frame_ids.add(tid)
                    boxes_list.append(box)
                    ids_list.append(tid)

            # قراءة اللوحة كل AI_INTERVAL
            if (now - last_ai_time) >= AI_INTERVAL:
                last_ai_time = now
                for box, tid in zip(boxes_list, ids_list):
                    x1, y1, x2, y2 = map(int, box)
                    crop = frame[
                        max(0, y1):min(frame.shape[0], y2),
                        max(0, x1):min(frame.shape[1], x2),
                    ]
                    if crop.size == 0:
                        continue

                    plate_text = read_plate(crop)
                    if plate_text:
                        plate_cache[tid] = plate_text
                        print(f"[✓] ID {tid} → {plate_text}")

            # تنظيف السيارات المختفية
            disappeared = set(plate_cache.keys()) - current_frame_ids
            for tid in disappeared:
                plate_cache.pop(tid, None)

            # رسم الـ bounding box والنمرة على الشاشة
            for box, tid in zip(boxes_list, ids_list):
                x1, y1, x2, y2 = map(int, box)
                plate = plate_cache.get(tid, "")
                color = (0, 255, 0) if plate else (0, 165, 255)

                cv2.rectangle(display, (x1, y1), (x2, y2), color, 2)

                label = f"ID {tid}"
                if plate:
                    label += f"  |  {plate}"

                # خلفية سوداء للنص عشان يبان
                (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
                cv2.rectangle(display, (x1, y1 - th - 10), (x1 + tw + 4, y1), (0, 0, 0), -1)
                cv2.putText(display, label, (x1 + 2, y1 - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

            cv2.imshow("Titan — Vehicle AI", display)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    finally:
        cap.release()
        cv2.destroyAllWindows()
        print("النظام أُغلق.")


if __name__ == "__main__":
    main()
