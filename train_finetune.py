"""
=============================================================================
  DAMON AI - KỊCH BẢN HUẤN LUYỆN BỔ SUNG (FINE-TUNING / CONTINUOUS LEARNING)
=============================================================================
Kịch bản này sử dụng tập dữ liệu được lưu từ tính năng "⚠️ Sửa nhãn" trên Web App
(hoặc tập dữ liệu tùy chỉnh) để huấn luyện lại mô hình YOLOv8 với kỹ thuật:
  1. Freeze Backbone (giữ lại tri thức nhận diện cũ, không bị quên)
  2. Learning rate nhỏ (tinh chỉnh chính xác các ca bị đoán sai như Thẻ vs Sách)
  3. Tự động nhận diện GPU NVIDIA CUDA hoặc CPU

Cách dùng:
  python train_finetune.py
  python train_finetune.py --epochs 30 --batch 8 --model yolov8s.pt
"""

import os
import sys
import argparse
import shutil
import torch
from ultralytics import YOLO

def main():
    parser = argparse.ArgumentParser(description="DAMON AI - Fine-tuning Script")
    parser.add_argument("--data", type=str, default="dataset/feedback/data.yaml", help="Đường dẫn đến file data.yaml")
    parser.add_argument("--model", type=str, default="yolov8s.pt", help="Mô hình nền (yolov8s.pt, yolov8m.pt hoặc yolov8s-worldv2.pt)")
    parser.add_argument("--epochs", type=int, default=25, help="Số vòng lặp huấn luyện (epochs)")
    parser.add_argument("--batch", type=int, default=8, help="Batch size")
    parser.add_argument("--imgsz", type=int, default=640, help="Kích thước ảnh")
    parser.add_argument("--freeze", type=int, default=10, help="Số tầng backbone đóng băng để giữ tri thức cũ")
    parser.add_argument("--lr", type=float, default=0.001, help="Tốc độ học khởi tạo (Learning Rate)")
    args = parser.parse_args()

    base_dir = os.path.dirname(os.path.abspath(__file__))
    yaml_path = os.path.join(base_dir, args.data) if not os.path.isabs(args.data) else args.data
    images_dir = os.path.join(os.path.dirname(yaml_path), "images")

    print("\n" + "=" * 65)
    print("      DAMON AI - HUẤN LUYỆN BỔ SUNG (FINE-TUNING YOLOv8)")
    print("=" * 65 + "\n")

    # 1. Kiểm tra tập dữ liệu
    if not os.path.exists(yaml_path):
        print(f"❌ Không tìm thấy file cấu hình dữ liệu: {yaml_path}")
        print("💡 GỢI Ý: Hãy mở Web App, quét đồ vật và bấm nút '⚠️ Sửa nhãn' để thu thập mẫu trước!")
        return

    sample_count = 0
    if os.path.exists(images_dir):
        sample_count = len([f for f in os.listdir(images_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))])

    print(f"📁 Tệp cấu hình: {yaml_path}")
    print(f"📸 Số mẫu ảnh đã thu thập: {sample_count} mẫu")

    if sample_count < 3:
        print("\n⚠️ CẢNH BÁO: Số lượng mẫu hiện tại khá ít (< 3 mẫu).")
        print("👉 Lời khuyên: Để mô hình học tốt và không bị nhầm, bạn nên chụp từ 5 - 15 tấm ảnh")
        print("   ở các góc khác nhau (cầm trên tay, đặt lên bàn, góc nghiêng/thẳng)...")
        print("   Bạn có thể mở Web App -> Đưa thẻ trước camera -> Bấm '⚠️ Sửa nhãn' vài lần.")
        confirm = input("\nBạn có muốn tiếp tục chạy thử nghiệm ngay không? (y/n): ").strip().lower()
        if confirm != 'y':
            print("Đã hủy quá trình huấn luyện.")
            return

    # 2. Kiểm tra thiết bị GPU / CPU
    if torch.cuda.is_available():
        device = 0
        gpu_name = torch.cuda.get_device_name(0)
        print(f"🚀 Thiết bị huấn luyện: GPU NVIDIA ({gpu_name})")
    else:
        device = "cpu"
        print("⚠️ Thiết bị huấn luyện: CPU (sẽ chậm hơn so với GPU)")

    print(f"🧠 Mô hình nền: {args.model}")
    print(f"🔄 Số epochs: {args.epochs} | Batch size: {args.batch} | Freeze: {args.freeze} layers")
    print("\n⏳ Bắt đầu huấn luyện...")

    try:
        model = YOLO(args.model)
        results = model.train(
            data=yaml_path,
            epochs=args.epochs,
            imgsz=args.imgsz,
            batch=args.batch,
            freeze=args.freeze,
            lr0=args.lr,
            device=device,
            project=os.path.join(base_dir, "runs", "finetune"),
            name="train_fix",
            exist_ok=True,
            verbose=True
        )

        # Tìm đường dẫn file best.pt linh hoạt (do Ultralytics có thể lồng thư mục theo task)
        candidate_paths = [
            str(getattr(getattr(model, "trainer", None), "best", "")),
            os.path.join(str(getattr(results, "save_dir", "")), "weights", "best.pt"),
            os.path.join(base_dir, "runs", "finetune", "train_fix", "weights", "best.pt"),
            os.path.join(base_dir, "runs", "detect", "runs", "finetune", "train_fix", "weights", "best.pt"),
            os.path.join(base_dir, "runs", "detect", "train_fix", "weights", "best.pt")
        ]

        found_weight = None
        for p in candidate_paths:
            if p and os.path.exists(p) and os.path.isfile(p):
                found_weight = p
                break

        target_weight = os.path.join(base_dir, "weights", "best_finetuned.pt")

        if found_weight:
            os.makedirs(os.path.dirname(target_weight), exist_ok=True)
            shutil.copy(found_weight, target_weight)
            print("\n" + "=" * 65)
            print("🎉 HUẤN LUYỆN THÀNH CÔNG RỰC RỠ!")
            print(f"✅ Đã tìm thấy tệp trọng số tại: {found_weight}")
            print(f"✅ Đã sao chép vào thư mục weights: {target_weight}")
            print("=" * 65)
            print("👉 Bây giờ bạn có thể nạp trọng số mới này vào hệ thống:")
            print("   Mở app.py - mô hình Hybrid sẽ tự động nạp 'weights/best_finetuned.pt'")
            print("=================================================================\n")
        else:
            print("\n⚠️ Huấn luyện hoàn thành nhưng không tìm thấy file best.pt đầu ra.")
            print("   Các đường dẫn đã quét:")
            for p in candidate_paths:
                if p:
                    print(f"   - {p}")

    except Exception as e:
        print(f"\n❌ Đã xảy ra lỗi trong quá trình huấn luyện: {e}")

if __name__ == "__main__":
    main()
