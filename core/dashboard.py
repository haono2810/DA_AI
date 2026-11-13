"""
=============================================================================
  DAMON AI - GIAO DIỆN BẢNG ĐIỀU KHIỂN DESKTOP (OPENCV HUD DASHBOARD)
=============================================================================
Vẽ bảng điều khiển đồ họa hiển thị trực tiếp:
  - Khung hình camera kèm Bounding Box neon và Nhãn Tiếng Việt
  - Tốc độ khung hình thực tế (FPS)
  - Thông số mô hình AI, ngưỡng tự tin và trạng thái tối ưu ánh sáng
  - Danh sách lịch sử các đồ vật đã phát hiện
"""

import cv2
import numpy as np
from PIL import Image, ImageDraw

from .config import (
    DASHBOARD_WIDTH,
    DASHBOARD_HEIGHT,
    CAMERA_WIDTH,
    CAMERA_HEIGHT,
)
from .vision import fit_image, vietnamese_font


def draw_dashboard(camera_image, translation, detected_history, conf_threshold, auto_enhance, fps, model_name, learning_lang_label="Tiếng Việt 🇻🇳"):
    """Vẽ toàn bộ bảng điều khiển giao diện Desktop sang dạng BGR numpy array."""
    dashboard = np.full(
        (DASHBOARD_HEIGHT, DASHBOARD_WIDTH, 3), (14, 18, 24), dtype=np.uint8
    )

    camera_panel = fit_image(camera_image, CAMERA_WIDTH, CAMERA_HEIGHT)
    dashboard[95:95 + CAMERA_HEIGHT, 32:32 + CAMERA_WIDTH] = camera_panel
    cv2.rectangle(dashboard, (32, 95), (32 + CAMERA_WIDTH, 95 + CAMERA_HEIGHT), (50, 64, 82), 2)

    list_left = 930
    divider_x = 910
    cv2.line(dashboard, (divider_x, 95), (divider_x, DASHBOARD_HEIGHT - 32), (40, 52, 68), 1)

    dashboard_rgb = cv2.cvtColor(dashboard, cv2.COLOR_BGR2RGB)
    canvas = Image.fromarray(dashboard_rgb)
    draw = ImageDraw.Draw(canvas)

    title_font = vietnamese_font(26)
    subtitle_font = vietnamese_font(15)
    label_font = vietnamese_font(16)
    value_font = vietnamese_font(25)
    list_font = vietnamese_font(18)
    hotkey_font = vietnamese_font(14)

    draw.text((32, 16), "HỆ THỐNG NHẬN DIỆN ĐỒ VẬT 120HZ ULTRA-SMOOTH", font=title_font, fill=(240, 246, 252))
    status_str = f"Tốc độ: {fps:.0f} FPS  |  Mô hình: {model_name} [M]  |  Học: {learning_lang_label} [L]  |  Độ tin cậy: {conf_threshold:.0%} [+/-]  |  Sáng: {'BẬT [E]' if auto_enhance else 'TẮT [E]'}"
    draw.text((32, 54), status_str, font=subtitle_font, fill=(80, 220, 160))

    draw.text((32, 630), "VẬT THỂ CHÍNH NỔI BẬT KHUNG HÌNH", font=label_font, fill=(130, 148, 170))
    draw.text(
        (32, 665), translation if translation else "Chưa phát hiện đồ vật...",
        font=value_font, fill=(90, 230, 160),
    )

    help_str = "Phím tắt:  [M] Đổi Mô Hình   [L] Đổi Ngôn Ngữ Học   [+] Tăng tự tin   [-] Giảm tự tin   [E] Bật/Tắt sáng   [R] Xóa lịch sử   [Q] Thoát"
    draw.text((32, 730), help_str, font=hotkey_font, fill=(150, 168, 190))

    draw.text((list_left, 95), f"ĐỒ VẬT ĐÃ PHÁT HIỆN ({len(detected_history)})", font=label_font, fill=(240, 246, 252))

    for index, (primary_key, vi_name) in enumerate(sorted(detected_history.items())):
        item_top = 140 + index * 38
        if item_top > DASHBOARD_HEIGHT - 45:
            break
        draw.text(
            (list_left, item_top), f"• {primary_key.capitalize()}: {vi_name}",
            font=list_font, fill=(215, 228, 240),
        )

    return cv2.cvtColor(np.asarray(canvas), cv2.COLOR_RGB2BGR)
