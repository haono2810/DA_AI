"""
=============================================================================
  DAMON AI - BỘ ĐỘNG CƠ NHẬN DIỆN VẬT THỂ (YOLO DETECTOR ENGINE)
=============================================================================
Đóng gói logic suy luận mạng Nơ-ron YOLOv8 / YOLOWorld:
  - Quản lý nạp mô hình đa phiên bản (Small, Large, Finetuned)
  - Dự đoán đồng bộ (Sync API cho Web App)
  - Dự đoán bất đồng bộ đa luồng (Async Engine cho Desktop 120Hz)
  - Bộ lọc nội suy làm mượt tọa độ Bounding Box (SmoothTracker)
"""

import os
import time
import threading
import numpy as np
import torch
from ultralytics import YOLOWorld, YOLO

from .config import (
    DEVICE,
    OBJECT_DEFINITIONS,
    ALL_PROMPTS,
    PROMPT_TO_PRIMARY,
    CONF_THRESHOLD,
    NEAR_OBJECT_MIN_AREA_RATIO,
    MODEL_SMALL_PATH,
    MODEL_LARGE_PATH,
    MODEL_FINETUNED_PATH,
)
from .vision import enhance_image


# ==================== HÀM TIỆN ÍCH DÙNG CHUNG ====================

def compute_iou(box1, box2):
    """Tính Intersection-over-Union (IoU) giữa 2 bounding box [x1, y1, x2, y2]."""
    xA, yA = max(box1[0], box2[0]), max(box1[1], box2[1])
    xB, yB = min(box1[2], box2[2]), min(box1[3], box2[3])
    inter_area = max(0, xB - xA) * max(0, yB - yA)
    area1 = max(0, box1[2] - box1[0]) * max(0, box1[3] - box1[1])
    area2 = max(0, box2[2] - box2[0]) * max(0, box2[3] - box2[1])
    union_area = area1 + area2 - inter_area
    return inter_area / union_area if union_area > 0 else 0.0


def fuse_detections(priority_dets, secondary_dets, iou_threshold=0.30):
    """
    Kết hợp 2 danh sách nhận diện theo cơ chế Hybrid Priority:
    - Giữ lại toàn bộ kết quả từ mô hình Finetuned.
    - Lọc bỏ các box của mô hình gốc nếu bị trùng vị trí (IoU >= threshold).
    - Giữ lại các vật thể khác mà mô hình gốc phát hiện được.
    """
    if not priority_dets:
        return list(secondary_dets)
    if not secondary_dets:
        return list(priority_dets)

    fused = list(priority_dets)
    for sec in secondary_dets:
        if not any(compute_iou(sec["box"], pri["box"]) >= iou_threshold for pri in priority_dets):
            fused.append(sec)
    return fused


def extract_detections(model, frame, conf, device, max_det=8):
    """
    Hàm dùng chung: chạy suy luận YOLO trên một mô hình cụ thể,
    trả về danh sách dict chứa box, primary, vi/en/zh/ko/ru, conf.
    """
    if model is None or frame is None:
        return []

    with torch.inference_mode():
        results = model.predict(
            frame, conf=conf, imgsz=640, iou=0.45,
            max_det=max_det, device=device, verbose=False,
        )

    boxes = results[0].boxes
    if boxes is None or len(boxes) == 0:
        return []

    h, w = frame.shape[:2]
    frame_area = w * h
    detections = []

    for box in boxes:
        x1, y1, x2, y2 = box.xyxy[0].tolist()
        box_area = max(0, x2 - x1) * max(0, y2 - y1)
        if box_area / frame_area < NEAR_OBJECT_MIN_AREA_RATIO:
            continue

        cls_id = int(box.cls.item())
        if hasattr(model, "names") and cls_id in model.names:
            raw_name = model.names[cls_id]
        elif cls_id < len(ALL_PROMPTS):
            raw_name = ALL_PROMPTS[cls_id]
        else:
            raw_name = f"object_{cls_id}"

        primary = PROMPT_TO_PRIMARY.get(raw_name, raw_name)
        obj_def = OBJECT_DEFINITIONS.get(primary, {})

        detections.append({
            "box": [int(x1), int(y1), int(x2), int(y2)],
            "primary": primary,
            "vi": obj_def.get("vi", primary),
            "en": obj_def.get("en", primary),
            "zh": obj_def.get("zh", primary),
            "ko": obj_def.get("ko", primary),
            "ru": obj_def.get("ru", primary),
            "conf": float(box.conf.item()),
        })

    return detections


def _load_yolo_model(path, device, is_world=True):
    """Nạp mô hình YOLOWorld hoặc YOLO chuẩn và đưa lên thiết bị phù hợp."""
    model = YOLOWorld(path) if is_world else YOLO(path)
    if is_world:
        model.set_classes(ALL_PROMPTS)
    if device == 0:
        model.to("cuda")
    return model


# ==================== DETECTOR ĐỒNG BỘ (WEB APP) ====================

class Detector:
    """Quản lý và thực thi suy luận mô hình YOLO cho Web App & Backend."""

    def __init__(self, device=DEVICE):
        self.device = device
        self.models = {}
        self.finetuned_mtime = 0
        self._load_default_models()

    def _load_default_models(self):
        for name, path in [("small", MODEL_SMALL_PATH), ("large", MODEL_LARGE_PATH)]:
            print(f"-> Đang tải mô hình YOLOv8-{name.capitalize()}...")
            try:
                self.models[name] = _load_yolo_model(path, self.device, is_world=True)
            except Exception as e:
                print(f"⚠️ Cảnh báo tải model {name}: {e}")
        self._load_finetuned_model()

    def _load_finetuned_model(self):
        """Nạp hoặc nạp lại mô hình đã fine-tune khi file trên đĩa được cập nhật."""
        if not os.path.exists(MODEL_FINETUNED_PATH):
            if "finetuned" in self.models:
                del self.models["finetuned"]
                self.finetuned_mtime = 0
                print("ℹ️ Mô hình finetuned đã được gỡ khỏi bộ nhớ (file không còn tồn tại).")
            return

        try:
            mtime = os.path.getmtime(MODEL_FINETUNED_PATH)
            if mtime > self.finetuned_mtime:
                print("-> Đang nạp mô hình đã Huấn luyện bổ sung (weights/best_finetuned.pt)...")
                self.models["finetuned"] = _load_yolo_model(MODEL_FINETUNED_PATH, self.device, is_world=False)
                self.finetuned_mtime = mtime
                print("✅ Nạp thành công mô hình Finetuned!")
        except Exception as e:
            print(f"⚠️ Không thể tải model finetuned: {e}")

    def predict(self, frame, conf=CONF_THRESHOLD, model_name="hybrid", enhance=False, max_det=8):
        """
        Dự đoán vật thể trên khung hình ảnh BGR.
        Hỗ trợ chế độ 'hybrid' (kết hợp Finetuned + YOLO-World) hoặc đơn lẻ.
        """
        if frame is None:
            return []
        if enhance:
            frame = enhance_image(frame)

        self._load_finetuned_model()

        if model_name == "hybrid":
            finetuned = self.models.get("finetuned")
            base = self.models.get("small")
            fine_dets = extract_detections(finetuned, frame, max(0.20, conf - 0.05), self.device, max_det) if finetuned else []
            base_dets = extract_detections(base, frame, conf, self.device, max_det) if base else []
            return fuse_detections(fine_dets, base_dets)

        selected = self.models.get(model_name) or self.models.get("small")
        return extract_detections(selected, frame, conf, self.device, max_det)


# ==================== DETECTOR BẤT ĐỒNG BỘ (DESKTOP 120HZ) ====================

class AsyncDetector:
    """Xử lý mô hình AI trên luồng riêng biệt để giao diện Desktop không bị nghẽn giật."""

    def __init__(self, model_path="hybrid", device=DEVICE):
        self.device = device
        self.mode = "hybrid" if model_path == "hybrid" else "single"

        base_path = MODEL_SMALL_PATH if model_path in ("hybrid", MODEL_SMALL_PATH) else model_path
        is_world = "best_finetuned" not in base_path
        self.model = _load_yolo_model(base_path, device, is_world=is_world)

        self.finetuned_model = None
        self._load_finetuned()

        self.latest_frame = None
        self.latest_detections = []
        self.conf_thresh = CONF_THRESHOLD
        self.running = True
        self.lock = threading.Lock()
        self.new_frame_event = threading.Event()

        self.thread = threading.Thread(target=self._inference_loop, daemon=True)
        self.thread.start()

    def _load_finetuned(self):
        if os.path.exists(MODEL_FINETUNED_PATH):
            try:
                self.finetuned_model = _load_yolo_model(MODEL_FINETUNED_PATH, self.device, is_world=False)
                print("✅ AsyncDetector đã nạp mô hình Finetuned!")
            except Exception as e:
                print(f"⚠️ AsyncDetector không thể nạp finetuned: {e}")

    def push_frame(self, frame, conf_thresh):
        with self.lock:
            self.latest_frame = frame
            self.conf_thresh = conf_thresh
        self.new_frame_event.set()

    def switch_model(self, new_mode_or_path):
        with self.lock:
            if new_mode_or_path == "hybrid":
                self.mode = "hybrid"
                self.model = _load_yolo_model(MODEL_SMALL_PATH, self.device, is_world=True)
                self._load_finetuned()
            else:
                self.mode = "single"
                is_world = "best_finetuned" not in new_mode_or_path
                self.model = _load_yolo_model(new_mode_or_path, self.device, is_world=is_world)

    def get_detections(self):
        with self.lock:
            return list(self.latest_detections)

    def _inference_loop(self):
        while self.running:
            if not self.new_frame_event.wait(timeout=0.05):
                continue
            self.new_frame_event.clear()

            with self.lock:
                frame, conf, mode = self.latest_frame, self.conf_thresh, self.mode

            if frame is None:
                continue

            if mode == "hybrid":
                fine_dets = extract_detections(self.finetuned_model, frame, max(0.20, conf - 0.05), self.device) if self.finetuned_model else []
                base_dets = extract_detections(self.model, frame, conf, self.device)
                final_dets = fuse_detections(fine_dets, base_dets)
            else:
                final_dets = extract_detections(self.model, frame, conf, self.device)

            with self.lock:
                self.latest_detections = final_dets

    def stop(self):
        self.running = False
        self.new_frame_event.set()
        if self.thread.is_alive():
            self.thread.join(timeout=0.5)


# ==================== BỘ LỌC LÀM MƯỢT BOUNDING BOX ====================

_LANG_KEYS = ("vi", "en", "zh", "ko", "ru")


class SmoothTracker:
    """Nội suy tọa độ khung hình (EMA Smoothing) giúp bounding box lướt siêu mượt 120Hz."""

    def __init__(self, alpha=0.35, hold_time=0.35):
        self.alpha = alpha
        self.hold_time = hold_time
        self.tracks = {}

    def _build_track(self, det, box, conf):
        """Tạo dict track từ detection, dùng chung cho cả track mới và cập nhật."""
        track = {"box": box, "conf": conf, "last_seen": time.time()}
        pkey = det["primary"]
        for lang in _LANG_KEYS:
            track[lang] = det.get(lang, pkey)
        return track

    def update(self, new_detections):
        now = time.time()
        updated = {}

        for det in new_detections:
            pkey = det["primary"]
            new_box = np.array(det["box"], dtype=np.float32)

            if pkey in self.tracks:
                prev = self.tracks[pkey]
                smoothed_box = self.alpha * new_box + (1.0 - self.alpha) * prev["box"]
                smoothed_conf = 0.7 * prev["conf"] + 0.3 * det["conf"]
                updated[pkey] = self._build_track(det, smoothed_box, smoothed_conf)
            else:
                updated[pkey] = self._build_track(det, new_box, det["conf"])

        # Giữ lại vết tránh hiện tượng chớp nháy (Flickering)
        for pkey, track in self.tracks.items():
            if pkey not in updated and now - track["last_seen"] < self.hold_time:
                updated[pkey] = track

        self.tracks = updated
        return self.tracks
