"""
DAMON AI - Core Package
Bộ thư viện xử lý thông minh: Nhận diện vật thể, Xử lý ảnh, Camera, Giọng nói TTS và Quản lý dữ liệu.
"""

from .config import (
    DEVICE,
    GPU_NAME,
    OBJECT_DEFINITIONS,
    SUPPORTED_LANGUAGES,
    ALL_PROMPTS,
    PROMPT_TO_PRIMARY,
    CONF_THRESHOLD,
    NEAR_OBJECT_MIN_AREA_RATIO,
    MAX_FRAME_WIDTH,
    MAX_FRAME_HEIGHT,
    CAMERA_FPS,
    USE_WEBCAM,
    PHONE_CAMERA_URL,
    get_local_ip,
)
from .vision import enhance_image, resize_frame, fit_image, vietnamese_font
from .detector import Detector, AsyncDetector, SmoothTracker
from .camera import CameraStream
from .tts import generate_neural_tts
from .feedback import FeedbackManager
from .dashboard import draw_dashboard
