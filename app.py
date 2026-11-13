"""
=============================================================================
  DAMON AI - MOBILE & WEB APPLICATION SERVER
=============================================================================
Máy chủ Flask cung cấp API nhận diện thời gian thực qua Camera trình duyệt,
hỗ trợ sửa nhãn Active Learning và phát âm song ngữ Neural Voice.
"""

import sys
import numpy as np
import cv2
from flask import Flask, request, jsonify, send_from_directory, Response

from core import (
    DEVICE,
    GPU_NAME,
    OBJECT_DEFINITIONS,
    SUPPORTED_LANGUAGES,
    get_local_ip,
    Detector,
    FeedbackManager,
    generate_neural_tts,
)

app = Flask(__name__, static_folder="static", static_url_path="/static")
app.config["MAX_CONTENT_LENGTH"] = 10 * 1024 * 1024  # Giới hạn upload 10MB

# Khởi tạo các dịch vụ Core
print(f"-> Thiết bị tính toán Web App: {'GPU NVIDIA (' + GPU_NAME + ')' if DEVICE == 0 else 'CPU'}")
detector = Detector(device=DEVICE)
feedback_mgr = FeedbackManager()


# ==================== ĐIỀU HƯỚNG GIAO DIỆN ====================

@app.route("/")
def index():
    """Giao diện chính Web App."""
    return send_from_directory("static", "index.html")


# ==================== CÁC API XỬ LÝ AI & DỮ LIỆU ====================

@app.route("/api/predict", methods=["POST"])
def predict():
    """Dự đoán vật thể trên khung hình tải lên từ camera trình duyệt."""
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]
    try:
        conf = float(request.form.get("conf", 0.35))
        conf = max(0.05, min(0.95, conf))
    except (ValueError, TypeError):
        conf = 0.35
    enhance = request.form.get("enhance", "false").lower() == "true"
    model_name = request.form.get("model", "hybrid")

    img_bytes = file.read()
    nparr = np.frombuffer(img_bytes, np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if frame is None:
        return jsonify({"error": "Invalid image"}), 400

    detections = detector.predict(
        frame=frame,
        conf=conf,
        model_name=model_name,
        enhance=enhance,
        max_det=8
    )

    return jsonify({
        "detections": detections,
        "gpu_name": GPU_NAME
    })


@app.route("/api/classes", methods=["GET"])
def get_classes():
    """Lấy danh sách các lớp vật thể hỗ trợ đa ngôn ngữ và số mẫu phản hồi đã thu thập."""
    lang_keys = ("vi", "en", "zh", "ko", "ru")
    items = [
        {"key": k, **{lang: v.get(lang, k) for lang in lang_keys}}
        for k, v in OBJECT_DEFINITIONS.items()
    ]
    return jsonify({
        "classes": items,
        "languages": SUPPORTED_LANGUAGES,
        "sample_count": feedback_mgr.count_samples(),
    })


@app.route("/api/feedback", methods=["POST"])
def submit_feedback():
    """Lưu trữ mẫu ảnh người dùng bấm 'Sửa nhãn' để huấn luyện lại mô hình."""
    if "file" not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    file = request.files["file"]
    predicted = request.form.get("predicted", "").strip()
    correct_key = request.form.get("correct_key", "").strip()
    correct_vi = request.form.get("correct_vi", "").strip()
    bbox_str = request.form.get("bbox", "")

    try:
        img_bytes = file.read()
        meta_entry, total_samples = feedback_mgr.save_feedback(
            img_bytes=img_bytes,
            predicted=predicted,
            correct_key=correct_key,
            correct_vi=correct_vi,
            bbox_str=bbox_str
        )
        print(f"-> [FEEDBACK] Đã lưu mẫu sửa nhãn #{total_samples}: '{predicted}' -> '{correct_key}' ({correct_vi})")

        return jsonify({
            "status": "success",
            "message": f"Đã lưu mẫu thành công! Tổng số mẫu: {total_samples}",
            "sample_id": meta_entry["id"],
            "total_samples": total_samples
        })
    except Exception as e:
        print(f"❌ Lỗi lưu feedback: {e}")
        return jsonify({"error": str(e)}), 400


@app.route("/api/tts")
def tts_route():
    """Sinh âm thanh giọng đọc Neural TTS cho từ vựng đa ngôn ngữ (Việt, Anh, Trung, Hàn, Nga)."""
    text = request.args.get("text", "").strip()
    lang = request.args.get("lang", "vi").strip().lower()
    sub_text = request.args.get("sub", "").strip()
    sub_lang = request.args.get("sub_lang", "en").strip().lower()

    # Tương thích ngược với các tham số cũ ?en=...&vi=...
    en_text = request.args.get("en", "").strip()
    vi_text = request.args.get("vi", "").strip()

    if not text and not en_text and not vi_text:
        return "Missing text", 400

    try:
        audio_data = generate_neural_tts(
            text=text or "",
            lang=lang,
            secondary_text=sub_text,
            secondary_lang=sub_lang,
            en_text=en_text,
            vi_text=vi_text,
        )
        if audio_data:
            return Response(audio_data, mimetype="audio/mpeg")
        return jsonify({"error": "TTS synthesis failed or offline"}), 502
    except Exception as e:
        print(f"⚠️ Lỗi phát âm Neural TTS (Cần 'pip install edge-tts'): {e}")
        return jsonify({"error": str(e)}), 500


# ==================== KHỞI ĐỘNG MÁY CHỦ ====================

if __name__ == "__main__":
    local_ip = get_local_ip()
    port = 5000
    use_http = "--http" in sys.argv

    if use_http:
        print("\n===============================================================")
        print("🚀 WEB APP DAMON AI ĐANG CHẠY CHẾ ĐỘ HTTP THƯỜNG (KHÔNG SSL)")
        print(f"👉 Trên máy tính truy cập: http://localhost:{port}")
        print(f"👉 Mở điện thoại truy cập: http://{local_ip}:{port}")
        print("💡 Lưu ý: Trên Chrome/Edge máy tính, Camera hoạt động bình thường trên http://localhost.")
        print("   (Nếu muốn dùng Camera trên điện thoại qua Wi-Fi, hãy chạy lại không có --http để bật HTTPS)")
        print("===============================================================\n")
        app.run(host="0.0.0.0", port=port, debug=False)
    else:
        print("\n===============================================================")
        print("🚀 MOBILE WEB APP DAMON AI ĐÃ BẬT HTTPS ĐỂ CẤP QUYỀN CAMERA!")
        print(f"👉 Trên máy tính truy cập: https://localhost:{port}")
        print(f"👉 Mở điện thoại (cùng mạng Wi-Fi) truy cập: https://{local_ip}:{port}")
        print("   📌 LƯU Ý KHI TRUY CẬP (DO DÙNG CHỨNG CHỈ TỰ KÝ ADHOC):")
        print("   Trình duyệt sẽ hiện cảnh báo bảo mật:")
        print("   -> Bấm 'Nâng cao' (Advanced) -> Bấm 'Tiếp tục truy cập' (Proceed) để vào web!")
        print("💡 Mẹo: Nếu chỉ dùng trên máy tính và muốn vào nhanh không cảnh báo, hãy gõ:")
        print("   python app.py --http")
        print("===============================================================\n")

        try:
            app.run(host="0.0.0.0", port=port, ssl_context="adhoc", debug=False)
        except Exception as e:
            print(f"-> Thử chạy HTTP thường do thiếu pyopenssl: {e}")
            app.run(host="0.0.0.0", port=port, debug=False)
