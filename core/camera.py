"""
=============================================================================
  DAMON AI - LUỒNG THU NHẬN HÌNH ẢNH CAMERA (THREADED CAMERA STREAM)
=============================================================================
Đọc khung hình từ Webcam hoặc IP Camera trên luồng nền (background thread),
loại bỏ hoàn toàn độ trễ I/O buffer để hiển thị tốc độ 60 - 120 FPS mượt mà.
"""

import time
import threading
import cv2


class CameraStream:
    """Đọc luồng camera ở luồng riêng biệt để đạt FPS hiển thị mượt mà không nghẽn."""

    def __init__(self, source, width, height, fps=60):
        self.cap = cv2.VideoCapture(source)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self.cap.set(cv2.CAP_PROP_FPS, fps)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        self.ret, self.frame = self.cap.read()
        self.running = True
        self.lock = threading.Lock()

        self.thread = threading.Thread(target=self._update, daemon=True)
        self.thread.start()

    def _update(self):
        while self.running:
            ret, frame = self.cap.read()
            if ret:
                with self.lock:
                    self.ret = ret
                    self.frame = frame
            else:
                time.sleep(0.005)

    def read(self):
        """Lấy bản sao khung hình mới nhất hiện tại."""
        with self.lock:
            return self.ret, (self.frame.copy() if self.frame is not None else None)

    def release(self):
        """Giải phóng thiết bị camera và dừng luồng thu hình."""
        self.running = False
        if self.thread.is_alive():
            self.thread.join(timeout=0.5)
        self.cap.release()
