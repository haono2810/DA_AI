"""
=============================================================================
  DAMON AI - CẤU HÌNH HỆ THỐNG & TỪ ĐIỂN ĐỒ VẬT
=============================================================================
Tệp này là nguồn dữ liệu chuẩn (Single Source of Truth) duy nhất cho toàn bộ dự án,
quản lý thiết bị GPU/CPU, danh mục vật thể Tiếng Việt, prompts nhận diện và các hằng số.
"""

import os
import socket
import torch

# ==================== CẤU HÌNH THIẾT BỊ PHẦN CỨNG ====================
DEVICE = 0 if torch.cuda.is_available() else "cpu"
GPU_NAME = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"

# ==================== ĐƯỜNG DẪN DỰ ÁN ====================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASET_DIR = os.path.join(BASE_DIR, "dataset")
FEEDBACK_DIR = os.path.join(DATASET_DIR, "feedback")
WEIGHTS_DIR = os.path.join(BASE_DIR, "weights")

# Mô hình mặc định
MODEL_SMALL_PATH = os.path.join(BASE_DIR, "yolov8s-worldv2.pt")
MODEL_LARGE_PATH = os.path.join(BASE_DIR, "yolov8l-worldv2.pt")
MODEL_FINETUNED_PATH = os.path.join(WEIGHTS_DIR, "best_finetuned.pt")

# ==================== HẰNG SỐ NHẬN DIỆN & CAMERA ====================
CONF_THRESHOLD = 0.35
NEAR_OBJECT_MIN_AREA_RATIO = 0.0005

MAX_FRAME_WIDTH = 640
MAX_FRAME_HEIGHT = 480
CAMERA_FPS = 60

DASHBOARD_WIDTH = 1350
DASHBOARD_HEIGHT = 780
CAMERA_WIDTH = 860
CAMERA_HEIGHT = 520

USE_WEBCAM = True
PHONE_CAMERA_URL = "http://172.17.57.222:8080/video"

# ==================== DANH MỤC NGÔN NGỮ HỖ TRỢ ====================
SUPPORTED_LANGUAGES = {
    "vi": {"name": "Tiếng Việt", "flag": "🇻🇳", "voice": "vi-VN-NamMinhNeural", "code": "vi-VN"},
    "en": {"name": "English", "flag": "🇬🇧", "voice": "en-US-GuyNeural", "code": "en-US"},
    "zh": {"name": "中文 (Trung)", "flag": "🇨🇳", "voice": "zh-CN-YunxiNeural", "code": "zh-CN"},
    "ko": {"name": "한국어 (Hàn)", "flag": "🇰🇷", "voice": "ko-KR-InJoonNeural", "code": "ko-KR"},
    "ru": {"name": "Русский (Nga)", "flag": "🇷🇺", "voice": "ru-RU-DmitryNeural", "code": "ru-RU"},
}

# ==================== DANH MỤC VẬT THỂ & PROMPTS CHUẨN XÁC ====================
OBJECT_DEFINITIONS = {
    "smartphone": {
        "vi": "Điện thoại",
        "en": "Smartphone",
        "zh": "手机",
        "ko": "스마트폰",
        "ru": "Смартфон",
        "prompts": ["smartphone", "mobile phone", "iphone"]
    },
    "laptop": {
        "vi": "Máy tính xách tay",
        "en": "Laptop",
        "zh": "笔记本电脑",
        "ko": "노트북",
        "ru": "Ноутбук",
        "prompts": ["laptop", "notebook computer"]
    },
    "mouse": {
        "vi": "Chuột máy tính",
        "en": "Mouse",
        "zh": "鼠标",
        "ko": "마우스",
        "ru": "Компьютерная мышь",
        "prompts": ["computer mouse"]
    },
    "keyboard": {
        "vi": "Bàn phím",
        "en": "Keyboard",
        "zh": "键盘",
        "ko": "키보드",
        "ru": "Клавиатура",
        "prompts": ["computer keyboard"]
    },
    "headset": {
        "vi": "Tai nghe",
        "en": "Headset",
        "zh": "耳机",
        "ko": "헤드셋 / 이어폰",
        "ru": "Наушники",
        "prompts": ["headset", "headphones", "earphones"]
    },
    "bottle": {
        "vi": "Cái chai",
        "en": "Bottle",
        "zh": "瓶子",
        "ko": "물병",
        "ru": "Бутылка",
        "prompts": ["water bottle", "plastic bottle", "bottle"]
    },
    "cup": {
        "vi": "Cái cốc / Tách",
        "en": "Cup / Mug",
        "zh": "水杯",
        "ko": "컵",
        "ru": "Чашка / Кружка",
        "prompts": ["cup", "mug", "coffee mug"]
    },
    "book": {
        "vi": "Quyển sách",
        "en": "Book",
        "zh": "书本",
        "ko": "책",
        "ru": "Книга",
        "prompts": ["reading book", "thick book", "book", "textbook", "notebook"]
    },
    "pen": {
        "vi": "Cây bút",
        "en": "Pen",
        "zh": "笔",
        "ko": "펜",
        "ru": "Ручка",
        "prompts": ["pen", "ballpoint pen", "writing pen", "gel pen", "fountain pen", "pencil"]
    },
    "glasses": {
        "vi": "Kính mắt",
        "en": "Glasses",
        "zh": "眼镜",
        "ko": "안경",
        "ru": "Очки",
        "prompts": ["eyeglasses", "sunglasses"]
    },
    "watch": {
        "vi": "Đồng hồ đeo tay",
        "en": "Wristwatch",
        "zh": "手表",
        "ko": "손목시계",
        "ru": "Наручные часы",
        "prompts": ["wristwatch", "smartwatch"]
    },
    "clock": {
        "vi": "Đồng hồ treo tường/bàn",
        "en": "Clock",
        "zh": "时钟",
        "ko": "시계",
        "ru": "Часы",
        "prompts": ["wall clock", "alarm clock"]
    },
    "remote": {
        "vi": "Điều khiển",
        "en": "Remote control",
        "zh": "遥控器",
        "ko": "리모컨",
        "ru": "Пульт ДУ",
        "prompts": ["remote control", "tv remote"]
    },
    "wallet": {
        "vi": "Ví tiền",
        "en": "Wallet",
        "zh": "钱包",
        "ko": "지갑",
        "ru": "Кошелек",
        "prompts": ["wallet", "purse"]
    },
    "key": {
        "vi": "Chìa khóa",
        "en": "Key",
        "zh": "钥匙",
        "ko": "열쇠",
        "ru": "Ключ",
        "prompts": ["key", "keys"]
    },
    "charger": {
        "vi": "Củ / Dây sạc",
        "en": "Charger",
        "zh": "充电器",
        "ko": "충전기",
        "ru": "Зарядное устройство",
        "prompts": ["phone charger", "charging cable", "power adapter", "charge"]
    },
    "chair": {
        "vi": "Cái ghế",
        "en": "Chair",
        "zh": "椅子",
        "ko": "의자",
        "ru": "Стул",
        "prompts": ["chair", "office chair"]
    },
    "backpack": {
        "vi": "Ba lô",
        "en": "Backpack",
        "zh": "背包",
        "ko": "배낭",
        "ru": "Рюкзак",
        "prompts": ["backpack", "schoolbag"]
    },
    "scissors": {
        "vi": "Cái kéo",
        "en": "Scissors",
        "zh": "剪刀",
        "ko": "가위",
        "ru": "Ножницы",
        "prompts": ["pair of scissors", "cutting scissors", "shears"]
    },
    "calculator": {
        "vi": "Máy tính cầm tay",
        "en": "Calculator",
        "zh": "计算器",
        "ko": "계산기",
        "ru": "Калькулятор",
        "prompts": ["calculator"]
    },
    "speaker": {
        "vi": "Loa âm thanh",
        "en": "Speaker",
        "zh": "音箱",
        "ko": "스피커",
        "ru": "Колонка",
        "prompts": ["speaker", "bluetooth speaker"]
    },
    "umbrella": {
        "vi": "Ô dù",
        "en": "Umbrella",
        "zh": "雨伞",
        "ko": "우산",
        "ru": "Зонт",
        "prompts": ["umbrella"]
    },
    "car": {
        "vi": "Ô tô",
        "en": "Car",
        "zh": "汽车",
        "ko": "자동차",
        "ru": "Автомобиль",
        "prompts": ["car", "automobile"]
    },
    "motorcycle": {
        "vi": "Xe máy",
        "en": "Motorcycle",
        "zh": "摩托车",
        "ko": "오토바이",
        "ru": "Мотоцикл",
        "prompts": ["motorcycle", "motorbike"]
    },
    "bicycle": {
        "vi": "Xe đạp",
        "en": "Bicycle",
        "zh": "自行车",
        "ko": "자전거",
        "ru": "Велосипед",
        "prompts": ["bicycle", "bike"]
    },
    "apple": {
        "vi": "Quả táo",
        "en": "Apple",
        "zh": "苹果",
        "ko": "사과",
        "ru": "Яблоко",
        "prompts": ["apple"]
    },
    "banana": {
        "vi": "Quả chuối",
        "en": "Banana",
        "zh": "香蕉",
        "ko": "바나나",
        "ru": "Банан",
        "prompts": ["banana"]
    },
    "spoon": {
        "vi": "Thìa / Muỗng",
        "en": "Spoon",
        "zh": "勺子",
        "ko": "숟가락",
        "ru": "Ложка",
        "prompts": ["spoon"]
    },
    "money": {
        "vi": "Tiền mặt",
        "en": "Banknote / Cash",
        "zh": "纸币 / 现金",
        "ko": "지폐 / 현금",
        "ru": "Деньги / Купюра",
        "prompts": ["paper money", "banknote"]
    },
    "helmet": {
        "vi": "Mũ bảo hiểm",
        "en": "Helmet",
        "zh": "头盔",
        "ko": "헬멧",
        "ru": "Шлем",
        "prompts": ["helmet"]
    },
    "tissue": {
        "vi": "Khăn giấy / Hộp giấy",
        "en": "Tissue paper",
        "zh": "纸巾",
        "ko": "휴지 / 티슈",
        "ru": "Салфетки",
        "prompts": ["tissue paper", "tissue box", "paper napkin", "facial tissue"]
    },
    "card": {
        "vi": "Thẻ / Card",
        "en": "Card",
        "zh": "卡片",
        "ko": "카드",
        "ru": "Карта / Карточка",
        "prompts": ["identity card", "id card", "credit card", "bank card", "atm card", "membership card", "plastic card", "driver license", "card"]
    },
    "power_outlet": {
        "vi": "Ổ cắm điện",
        "en": "Power outlet",
        "zh": "电源插座",
        "ko": "콘센트",
        "ru": "Розетка",
        "prompts": ["power outlet", "socket", "electrical socket", "power plug"]
    }
}

# Tạo danh sách Prompts phẳng và bảng tra cứu ngược
ALL_PROMPTS = []
PROMPT_TO_PRIMARY = {}
for primary_key, data in OBJECT_DEFINITIONS.items():
    for prompt in data["prompts"]:
        ALL_PROMPTS.append(prompt)
        PROMPT_TO_PRIMARY[prompt] = primary_key


def get_local_ip():
    """Lấy địa chỉ IPv4 nội bộ trong mạng LAN."""
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        try:
            s.connect(("8.8.8.8", 80))
            return s.getsockname()[0]
        finally:
            s.close()
    except Exception:
        return "127.0.0.1"
