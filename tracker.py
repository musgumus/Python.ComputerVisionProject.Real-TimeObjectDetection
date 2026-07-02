import os
import cv2
import json
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort

# ----------------- Constants ----------------- #

CLASS_NAMES = {
    3: "bus",
    1: "car",
    0: "truck",
    2: "van"
}

video_path = 'HD Stock Videos ｜ Urban ｜ Clear traffic [YfI-TN4BH4c].mp4'
model_path = 'runs/detect/train4/weights/best.pt'

pred_folder = "predictions"
pred_txt = os.path.join(pred_folder, "pred.txt")
pred_json = os.path.join(pred_folder, "pred_coco.json")

gt_txt = "predictions/gt.txt"  # Ground truth dosyanın tam yolu (sen ekle buraya)

# -----------------Preparation ----------------- #

os.makedirs(pred_folder, exist_ok=True)
for file in [pred_txt, pred_json]:
    if os.path.exists(file):
        os.remove(file)

model = YOLO(model_path)
tracker = DeepSort(max_age=30)
cap = cv2.VideoCapture(video_path)

coco_predictions = []
frame_idx = 0

# ----------------- Track and detection ----------------- #

while True:
    ret, frame = cap.read()
    if not ret:
        print("Video finished.")
        break

    frame_idx += 1
    results = model(frame, imgsz=640, conf=0.8)[0]

    detections = []
    if results.boxes and results.boxes.xyxy is not None:
        for box, conf, cls in zip(results.boxes.xyxy, results.boxes.conf, results.boxes.cls):
            x1, y1, x2, y2 = box.int().tolist()
            class_id = int(cls)
            confidence = float(conf)
            if confidence > 0.1:
                detections.append(([x1, y1, x2 - x1, y2 - y1], confidence, class_id))

    tracks = tracker.update_tracks(detections, frame=frame)
    active_ids = set()

    with open(pred_txt, "a") as f:
        for track in tracks:
            if not track.is_confirmed():
                continue

            track_id = track.track_id
            l, t, r, b = track.to_ltrb()
            x1, y1, x2, y2 = map(int, [l, t, r, b])
            conf = track.det_conf if track.det_conf is not None else 0.0
            class_id = track.det_class if hasattr(track, "det_class") else 0
            class_name = CLASS_NAMES.get(class_id, "unknown")

            active_ids.add(track_id)

            # MotChallenge formatı: frame, id, x, y, w, h, conf, class_id, visibility
            f.write(f"{frame_idx},{track_id},{x1},{y1},{x2 - x1},{y2 - y1},{conf:.2f},{class_id},1\n")

            coco_predictions.append({
                "image_id": frame_idx,
                "category_id": class_id,
                "bbox": [x1, y1, x2 - x1, y2 - y1],
                "score": float(conf)
            })

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f'ID {track_id} {class_name}', (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

    total_vehicles = len(active_ids)
    traffic_level = "LOW" if total_vehicles < 10 else "MID" if total_vehicles < 30 else "HIGH"

    cv2.putText(frame, f'Total vehicle: {total_vehicles}', (10, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
    cv2.putText(frame, f'Traffic Density: {traffic_level}', (10, 80),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    cv2.imshow("YOLO + DeepSORT Traffic Density", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()





