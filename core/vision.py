"""
=============================================================================
  DAMON AI - THƯ VIỆN XỬ LÝ ẢNH & HIỂN THỊ FONT (COMPUTER VISION)
=============================================================================
Cung cấp các thuật toán tối ưu hóa khung hình máy ảnh:
  - CLAHE + Hiệu chỉnh Gamma tương thích ánh sáng yếu / ngược sáng
  - Resize khung hình tối ưu hiệu năng
  - Tải Font chữ Tiếng Việt không lỗi hiển thị trên hệ điều hành Windows
"""

import cv2
import numpy as np
from PIL import ImageFont
from .config import MAX_FRAME_WIDTH, MAX_FRAME_HEIGHT


def enhance_image(frame):
    """
    Tự động cân bằng sáng và tăng độ tương phản chi tiết với CLAHE và Gamma động.
    Giúp YOLO nhận diện rõ nét ngay cả trong môi trường phòng tối hoặc đèn ngược sáng.
    """
    if frame is None:
        return None

    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    mean_l = np.mean(l)

    # 1. Tăng tương phản cục bộ qua CLAHE
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
    cl = clahe.apply(l)

    # 2. Điều chỉnh Gamma thích ứng theo độ sáng trung bình
    if mean_l < 75:      # Quá tối -> nâng sáng
        gamma = 1.35
    elif mean_l > 185:   # Quá lóa -> giảm chói
        gamma = 0.8
    else:
        gamma = 1.0

    if gamma != 1.0:
        inv_gamma = 1.0 / gamma
        table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
        cl = cv2.LUT(cl, table)

    enhanced_lab = cv2.merge((cl, a, b))
    return cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)


def resize_frame(frame, max_width=MAX_FRAME_WIDTH, max_height=MAX_FRAME_HEIGHT):
    """Co giãn kích thước khung hình giữ nguyên tỉ lệ để tăng tốc độ xử lý AI."""
    if frame is None:
        return None
    h, w = frame.shape[:2]
    scale = min(max_width / w, max_height / h)
    if scale >= 1.0:
        return frame
    new_w = max(1, int(w * scale))
    new_h = max(1, int(h * scale))
    return cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)


def fit_image(image, width, height, background=(20, 24, 32)):
    """Đặt ảnh vào khung chữ nhật theo đúng tỉ lệ gốc với viền nền đệm."""
    if image is None:
        return np.full((height, width, 3), background, dtype=np.uint8)

    image_height, image_width = image.shape[:2]
    scale = min(width / image_width, height / image_height)
    resized = cv2.resize(
        image,
        (max(1, int(image_width * scale)), max(1, int(image_height * scale))),
        interpolation=cv2.INTER_AREA,
    )
    panel = np.full((height, width, 3), background, dtype=np.uint8)
    offset_x = (width - resized.shape[1]) // 2
    offset_y = (height - resized.shape[0]) // 2
    panel[offset_y:offset_y + resized.shape[0], offset_x:offset_x + resized.shape[1]] = resized
    return panel


def vietnamese_font(size):
    """Tải font chữ TrueType hỗ trợ dấu Tiếng Việt, Trung, Hàn, Nga trên Windows."""
    font_paths = [
        "C:/Windows/Fonts/segoeui.ttf",
        "C:/Windows/Fonts/msyh.ttc",
        "C:/Windows/Fonts/malgun.ttf",
        "C:/Windows/Fonts/arial.ttf",
        "C:/Windows/Fonts/calibri.ttf",
        "C:/Windows/Fonts/tahoma.ttf",
    ]
    for path in font_paths:
        try:
            return ImageFont.truetype(path, size)
        except Exception:
            continue
    return ImageFont.load_default()
