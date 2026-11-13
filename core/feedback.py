"""
=============================================================================
  DAMON AI - QUẢN LÝ DỮ LIỆU PHẢN HỒI & HỌC LIÊN TỤC (ACTIVE LEARNING)
=============================================================================
Thu thập và lưu trữ các ca nhận diện cần sửa nhãn từ Web App để phục vụ
huấn luyện bổ sung (Fine-tuning):
  - Tự động chuẩn hóa tọa độ nhãn YOLO .txt [class_id, x_center, y_center, w, h]
  - Quản lý đồng bộ data.yaml, classes.json và annotations.json
"""

import os
import time
import json
import uuid
import threading
import cv2
import numpy as np

from .config import (
    FEEDBACK_DIR,
    OBJECT_DEFINITIONS,
)


class FeedbackManager:
    """Quản lý tập dữ liệu mẫu phản hồi sửa nhãn."""

    def __init__(self, feedback_dir=FEEDBACK_DIR):
        self.feedback_dir = feedback_dir
        self.images_dir = os.path.join(feedback_dir, "images")
        self.labels_dir = os.path.join(feedback_dir, "labels")
        self.yaml_file = os.path.join(feedback_dir, "data.yaml")
        self.classes_file = os.path.join(feedback_dir, "classes.json")
        self.annotations_file = os.path.join(feedback_dir, "annotations.json")
        self._lock = threading.Lock()

        os.makedirs(self.images_dir, exist_ok=True)
        os.makedirs(self.labels_dir, exist_ok=True)

        # Khởi tạo tệp cấu hình nếu chưa có
        self.load_classes()

    def update_yaml(self, classes):
        """Cập nhật tệp cấu hình data.yaml dùng cho YOLO huấn luyện."""
        clean_path = self.feedback_dir.replace("\\", "/")
        yaml_content = f"path: {clean_path}\ntrain: images\nval: images\n\nnames:\n"
        for idx, c in enumerate(classes):
            yaml_content += f"  {idx}: {c}\n"

        with open(self.yaml_file, "w", encoding="utf-8") as f:
            f.write(yaml_content)

    def save_classes(self, classes):
        """Lưu danh sách lớp vào classes.json và cập nhật yaml."""
        with open(self.classes_file, "w", encoding="utf-8") as f:
            json.dump(classes, f, ensure_ascii=False, indent=2)
        self.update_yaml(classes)

    def load_classes(self):
        """Tải danh sách các lớp đối tượng đang có trong tập feedback."""
        if os.path.exists(self.classes_file):
            try:
                with open(self.classes_file, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception:
                pass

        default_classes = list(OBJECT_DEFINITIONS.keys())
        self.save_classes(default_classes)
        return default_classes

    def count_samples(self):
        """Đếm tổng số ảnh mẫu hợp lệ đã thu thập."""
        if not os.path.exists(self.images_dir):
            return 0
        return len([f for f in os.listdir(self.images_dir) if f.lower().endswith((".jpg", ".jpeg", ".png"))])

    def save_feedback(self, img_bytes, predicted, correct_key, correct_vi="", bbox_str=""):
        """
        Lưu mẫu ảnh và nhãn chuẩn hóa YOLO.
        Trả về: (dict thông tin mẫu, tổng số mẫu)
        """
        if not correct_key:
            raise ValueError("Thiếu correct_key cho nhãn đúng.")

        current_classes = self.load_classes()
        if correct_key not in current_classes:
            current_classes.append(correct_key)
            self.save_classes(current_classes)

        class_id = current_classes.index(correct_key)

        nparr = np.frombuffer(img_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError("Không thể giải mã dữ liệu ảnh tải lên.")

        h, w = img.shape[:2]

        # Chuẩn hóa tọa độ Bounding box [0.0 - 1.0]
        x_center, y_center, bw, bh = 0.5, 0.5, 0.85, 0.85
        if bbox_str:
            try:
                bbox = json.loads(bbox_str) if isinstance(bbox_str, str) else bbox_str
                if len(bbox) == 4:
                    bx1, by1, bx2, by2 = bbox
                    if bx2 > 1.0 or by2 > 1.0:
                        bx1 = max(0, min(w, bx1)) / w
                        by1 = max(0, min(h, by1)) / h
                        bx2 = max(0, min(w, bx2)) / w
                        by2 = max(0, min(h, by2)) / h
                    x_center = (bx1 + bx2) / 2.0
                    y_center = (by1 + by2) / 2.0
                    bw = max(0.05, abs(bx2 - bx1))
                    bh = max(0.05, abs(by2 - by1))
            except Exception as e:
                print(f"⚠️ Lỗi giải mã bounding box: {e}")

        sample_id = f"{int(time.time())}_{uuid.uuid4().hex[:6]}"
        img_filename = f"{sample_id}.jpg"
        label_filename = f"{sample_id}.txt"

        img_path = os.path.join(self.images_dir, img_filename)
        label_path = os.path.join(self.labels_dir, label_filename)

        cv2.imwrite(img_path, img)

        with open(label_path, "w", encoding="utf-8") as f:
            f.write(f"{class_id} {x_center:.6f} {y_center:.6f} {bw:.6f} {bh:.6f}\n")

        meta_entry = {
            "id": sample_id,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "image": img_filename,
            "predicted": predicted,
            "correct_key": correct_key,
            "correct_vi": correct_vi or OBJECT_DEFINITIONS.get(correct_key, {}).get("vi", correct_key),
            "class_id": class_id,
            "bbox": [x_center, y_center, bw, bh]
        }

        meta_list = []
        with self._lock:
            if os.path.exists(self.annotations_file):
                try:
                    with open(self.annotations_file, "r", encoding="utf-8") as f:
                        meta_list = json.load(f)
                except Exception:
                    meta_list = []

            meta_list.append(meta_entry)
            with open(self.annotations_file, "w", encoding="utf-8") as f:
                json.dump(meta_list, f, ensure_ascii=False, indent=2)

        total_samples = self.count_samples()
        return meta_entry, total_samples
