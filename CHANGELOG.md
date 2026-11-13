# Changelog - DAMON AI

All notable changes to this project will be documented in this file.

## [1.4.1] - 2026-11-10

### 🔧 Sửa lỗi & Tối ưu hóa Logic Phát Âm và Đa Ngôn Ngữ
- **Khắc phục lỗi phát âm Tiếng Hàn (và lỗi tự động đọc kèm Tiếng Anh)**:
  - Loại bỏ hoàn toàn tham số ghép tiếng Anh `&sub=...&sub_lang=en` trong hàm `speakTargetWord` ở [`app.js`](file:///e:/KIEN_TRUC/NAM_4/KI_1/DO_AN_AI/DAMONAI/static/app.js#L306). Giờ đây hệ thống chỉ phát âm duy nhất từ vựng của ngôn ngữ đang học (Tiếng Hàn, Trung, Nga, Việt hoặc Anh).
  - Khắc phục lỗi `403 Invalid response status` của Microsoft Edge WebSocket bằng việc nâng cấp `edge-tts` lên phiên bản `7.3.1`.
  - Cập nhật giọng đọc chuẩn Hàn Quốc `ko-KR-SunHiNeural` (giọng tự nhiên, rõ chữ) và dự phòng `ko-KR-InJoonNeural`.
  - Loại bỏ việc chèn byte `\x00` thủ công trong luồng stream MP3 ở [`core/tts.py`](file:///e:/KIEN_TRUC/NAM_4/KI_1/DO_AN_AI/DAMONAI/core/tts.py#L41) để tránh làm hỏng định dạng khung MP3 trên các trình duyệt.
- **Tối ưu hóa phản hồi giao diện tức thì**:
  - Khi người dùng đổi ngôn ngữ từ menu chọn học (ví dụ: đổi sang Tiếng Hàn), hàm `updateLearningViews` sẽ cập nhật ngay lập tức banner từ vựng chính đang hiển thị trên màn hình mà không cần chờ khung hình tiếp theo.
  - Thêm phím tắt `[L]` trong ứng dụng Desktop [`demo.py`](file:///e:/KIEN_TRUC/NAM_4/KI_1/DO_AN_AI/DAMONAI/demo.py#L182) để duyệt qua các ngôn ngữ học tập (`vi`, `zh`, `ko`, `ru`, `en`) đồng bộ hoàn toàn với giao diện Web.
  - Tăng phiên bản cache-busting tài nguyên tĩnh lên `?v=2.7` trong [`index.html`](file:///e:/KIEN_TRUC/NAM_4/KI_1/DO_AN_AI/DAMONAI/static/index.html#L16).

## [1.4.0] - 2026-11-10

### 🌐 Tính năng mới: Chế độ Học Từ Vựng Đa Ngôn Ngữ (Flashcard AI Learning)
- **Tối ưu trải nghiệm học ngoại ngữ theo yêu cầu**:
  - Giao diện điều khiển (nút bấm, cài đặt, hướng dẫn, modal) được giữ nguyên **100% Tiếng Việt** để người dùng thao tác dễ dàng nhất.
  - Khi chọn ngôn ngữ muốn học (Trung 🇨🇳, Hàn 🇰🇷, Nga 🇷🇺, Anh 🇬🇧), hệ thống tự động hiển thị song song **[Từ vựng mục tiêu] • [Nghĩa Tiếng Việt]** trên khung Bounding Box, Banner quét và Lịch sử đồ vật.
  - Nút **🔊 Nghe đọc** và tính năng **Auto Đọc** sẽ phát âm chuẩn xác từ vựng ngoại ngữ đó bằng giọng đọc bản địa chất lượng cao (Microsoft Edge Neural Voice).
- **Từ điển đồ vật đa ngôn ngữ (`OBJECT_DEFINITIONS`)**:
  - Bổ sung bản dịch chính xác cho toàn bộ 33 danh mục vật thể hỗ trợ 5 ngôn ngữ:
    - 🇻🇳 Tiếng Việt (`vi`)
    - 🇬🇧 Tiếng Anh (`en`)
    - 🇨🇳 Tiếng Trung (`zh`)
    - 🇰🇷 Tiếng Hàn (`ko`)
    - 🇷🇺 Tiếng Nga (`ru`)
- **Bộ chuyển đổi ngôn ngữ trên giao diện Web (`index.html`, `style.css`, `app.js`)**:
  - Bổ sung menu lựa chọn ngôn ngữ trực quan trên thanh Header với lá cờ quốc gia và hiệu ứng Glassmorphism Cyberpunk.
  - Tự động ghi nhớ lựa chọn ngôn ngữ của người dùng qua `localStorage` (`damon_selected_lang`).
  - Toàn bộ nhãn bounding box, banner camera, lịch sử quét và kết quả tìm kiếm modal sửa nhãn tự động đổi ngôn ngữ tức thì.
- **Nâng cấp công cụ phát âm giọng đọc đa ngữ Neural Voice (`core/tts.py` & `/api/tts`)**:
  - Tích hợp giọng đọc Microsoft Edge Neural Voice bản địa cho từng ngôn ngữ:
    - Tiếng Việt: `vi-VN-NamMinhNeural`
    - Tiếng Anh: `en-US-GuyNeural`
    - Tiếng Trung: `zh-CN-YunxiNeural`
    - Tiếng Hàn: `ko-KR-InJoonNeural`
    - Tiếng Nga: `ru-RU-DmitryNeural`
  - Tự động fallback Web Speech API trên trình duyệt phù hợp từng mã ngôn ngữ (`zh-CN`, `ko-KR`, `ru-RU`, `vi-VN`, `en-US`).
- **Nâng cấp API backend (`app.py` & `core/detector.py`)**:
  - `/api/classes`: Trả về danh sách ngôn ngữ hỗ trợ `languages` và bản dịch 5 thứ tiếng cho từng class.
  - `/api/tts`: Hỗ trợ tham số `text`, `lang`, `sub`, `sub_lang` và giữ tương thích ngược với `en`, `vi`.
  - `Detector` & `AsyncDetector`: Trả về đầy đủ các trường `vi`, `en`, `zh`, `ko`, `ru` trong kết quả dự đoán.


## [1.3.0] - 2026-10-27

### 🚀 Tính năng mới: Chế độ Mô hình Lai (Hybrid AI Fusion)
- **Tích hợp bộ suy luận lai đa mô hình (`Hybrid Fusion`)**:
  - Chạy kết hợp song song giữa **Mô hình Đã Học (weights/best_finetuned.pt)** và **Mô hình Gốc (YOLO-World)**.
  - Thuật toán `fuse_detections` dựa trên tỷ lệ giao diện hộp `compute_iou >= 0.30`: Tự động ưu tiên nhãn do người dùng dạy (Cây bút, Kính mắt, Thẻ...), loại bỏ hoàn toàn hiện tượng đoán nhầm của mô hình gốc, đồng thời vẫn nhận diện đầy đủ tất cả các vật thể xung quanh.
- **Cập nhật giao diện**:
  - Web App: Thêm nút chọn **`🚀 Lai (Hybrid)`** được kích hoạt mặc định.
  - Desktop App: Bổ sung chế độ **`🚀 Hybrid AI`** vào vòng lặp chuyển đổi phím `[M]`.

## [1.2.0] - 2026-10-27

### 🏗️ Kiến trúc & Tái cấu trúc Module (Modularization Refactor)
- **Tạo gói `core/` tập trung**:
  - `core/config.py`: Nguồn dữ liệu cấu hình duy nhất (Single Source of Truth) cho 34 vật thể (`OBJECT_DEFINITIONS`), prompts, cấu hình GPU/CPU, ngưỡng tự tin và độ phân giải camera.
  - `core/vision.py`: Tách riêng các hàm xử lý thị giác máy tính: tăng cường ánh sáng thích ứng CLAHE + Gamma, co giãn khung hình và tải font Tiếng Việt Windows.
  - `core/detector.py`: Đóng gói bộ máy suy luận YOLO `Detector`, luồng suy luận bất đồng bộ `AsyncDetector`, và thuật toán lọc tọa độ `SmoothTracker` (EMA Smoothing).
  - `core/camera.py`: Đóng gói luồng đọc camera đa luồng `CameraStream` đạt 60-120 FPS.
  - `core/tts.py`: Đóng gói dịch vụ tổng hợp giọng đọc song ngữ chất lượng cao Microsoft Edge Neural Voice.
  - `core/feedback.py`: Quản lý quy trình Active Learning thu thập dữ liệu feedback và tự động cập nhật nhãn chuẩn YOLO.
  - `core/dashboard.py`: Tách giao diện bảng điều khiển đồ họa OpenCV Dark Mode cho desktop.
- **Rút gọn các tệp chính**:
  - `app.py`: Rút ngắn từ >500 dòng xuống ~130 dòng, tập trung hoàn toàn vào xử lý routing Flask và API.
  - `demo.py`: Rút ngắn từ >620 dòng xuống ~150 dòng, tách biệt luồng logic chính và giao diện.
- **Cải tiến huấn luyện**:
  - `train_finetune.py`: Tự động tìm kiếm file trọng số `best.pt` của Ultralytics và sao chép về `weights/best_finetuned.pt`.
