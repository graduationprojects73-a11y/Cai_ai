import cv2
from ultralytics import YOLO
import time
import requests
import threading
from concurrent.futures import ThreadPoolExecutor

LOCAL_API_PROCESS = "http://127.0.0.1:7860/process"
LOCAL_API_REMOVE  = "http://127.0.0.1:7860/remove_vehicle"

AI_INTERVAL = 0.5
MAX_WORKERS  = 4

vehicle_model = YOLO("yolo11n.pt")

# ── shared state ──────────────────────────────────────────────
_lock                = threading.Lock()
processed_ids        = set()   # سيارات تم تأكيد نمرتها
results_cache        = {}      # track_id -> plate string
in_flight_ids        = set()   # طلبات جارية الآن في الـ API
# ─────────────────────────────────────────────────────────────

executor = ThreadPoolExecutor(max_workers=MAX_WORKERS)


def _open_camera():
    for index in range(3):
        cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)
        if cap.isOpened():
            print(f"[Camera] فُتحت على index {index}")
            return cap
    raise RuntimeError("فشل فتح الكاميرا — تأكد من توصيل كاميرا وشغّل البرنامج مرة أخرى")


def send_to_api(track_id: int, crop):
    try:
        _, img_encoded = cv2.imencode('.png', crop)
        files = {'file': ('car.png', img_encoded.tobytes(), 'image/png')}
        data  = {'track_id': str(track_id)}

        response = requests.post(LOCAL_API_PROCESS, data=data, files=files, timeout=5)
        if response.status_code == 200:
            result = response.json()
            if result.get("status") == "Confirmed":
                plate = result.get("plate", "")
                with _lock:
                    processed_ids.add(track_id)
                    results_cache[track_id] = plate
                print(f"[✓] ID {track_id} | النمرة: {plate}")
    except Exception:
        pass
    finally:
        with _lock:
            in_flight_ids.discard(track_id)


def notify_remove(track_id: int):
    try:
        requests.post(LOCAL_API_REMOVE, data={'track_id': str(track_id)}, timeout=2)
    except Exception:
        pass


def main():
    cap = _open_camera()

    currently_tracked = set()
    last_ai_time      = 0.0

    print("النظام شغّال — اضغط Q للإيقاف")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("[!] انقطع feed الكاميرا")
                break

            now          = time.time()
            display      = frame.copy()

            tracks = vehicle_model.track(
                frame,
                persist=True,
                verbose=False,
                classes=[2, 3, 5, 7],
                tracker="bytetrack.yaml",
            )

            # ── اجمع IDs الـ frame الحالي دايماً ────────────────
            current_frame_ids = set()
            boxes_list        = []
            ids_list          = []

            if tracks[0].boxes.id is not None:
                raw_boxes = tracks[0].boxes.xyxy.cpu().numpy()
                raw_ids   = tracks[0].boxes.id.int().cpu().tolist()
                for box, tid in zip(raw_boxes, raw_ids):
                    current_frame_ids.add(tid)
                    boxes_list.append(box)
                    ids_list.append(tid)

            # ── إرسال للـ API كل AI_INTERVAL ────────────────────
            if (now - last_ai_time) >= AI_INTERVAL:
                last_ai_time = now
                for box, tid in zip(boxes_list, ids_list):
                    with _lock:
                        skip = (tid in processed_ids) or (tid in in_flight_ids)
                    if skip:
                        continue

                    x1, y1, x2, y2 = map(int, box)
                    crop = frame[
                        max(0, y1):min(frame.shape[0], y2),
                        max(0, x1):min(frame.shape[1], x2),
                    ]
                    if crop.size == 0:
                        continue

                    with _lock:
                        in_flight_ids.add(tid)

                    executor.submit(send_to_api, tid, crop.copy())

            # ── تنظيف السيارات المختفية ──────────────────────────
            disappeared = currently_tracked - current_frame_ids
            for tid in disappeared:
                executor.submit(notify_remove, tid)
                with _lock:
                    processed_ids.discard(tid)
                    results_cache.pop(tid, None)
                    in_flight_ids.discard(tid)

            currently_tracked = current_frame_ids

            # ── رسم على الشاشة ───────────────────────────────────
            for box, tid in zip(boxes_list, ids_list):
                x1, y1, x2, y2 = map(int, box)
                with _lock:
                    confirmed = tid in processed_ids
                    plate     = results_cache.get(tid, "")

                color = (0, 255, 0) if confirmed else (0, 165, 255)
                cv2.rectangle(display, (x1, y1), (x2, y2), color, 2)

                label = f"ID {tid}"
                if plate:
                    label += f" | {plate}"
                cv2.putText(display, label, (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

            cv2.imshow("Titan — Vehicle AI", display)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    finally:
        cap.release()
        cv2.destroyAllWindows()
        executor.shutdown(wait=False)
        print("النظام أُغلق.")


if __name__ == "__main__":
    main()
