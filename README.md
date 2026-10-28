# Lab 2 — Nhận diện cảm xúc khuôn mặt bằng Deep Learning

Project hoàn thiện theo đề Lab 2: FER-2013, CNN tự thiết kế (M1), ResNet18 Transfer Learning (M2), đánh giá M3–M5 và ứng dụng OpenCV nhận diện cảm xúc từ camera/ảnh/video. Ngôn ngữ hướng dẫn: tiếng Việt.

## 1. Cấu trúc

```text
Lab2_Emotion_Recognition/
├── README.md
├── requirements.txt
├── config.yaml
├── data/fer2013/{train,test}/...
├── notebooks/train.ipynb
├── src/model.py
├── src/train.py
├── src/detector.py
├── src/predict.py
├── src/app.py
├── weights/{cnn_best,transfer_best}.pth
├── results/{cnn,transfer}/...
└── report.pdf
```

## 2. Cài đặt (Python 3.9+)

Chạy lệnh từ thư mục gốc project:

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
```

FER-2013 đã được đặt tại `data/fer2013/`. Nếu bạn tải riêng dataset, cấu trúc cần có `data/fer2013/train/<class>/` và `data/fer2013/test/<class>/`. Chỉ sử dụng FER-2013, gồm 7 lớp: angry, disgust, fear, happy, neutral, sad, surprise.

## 3. Kết quả huấn luyện có sẵn

Đã tích hợp checkpoint và kết quả xuất từ Google Colab; không cần train lại để chạy demo.

| Mô hình | Test Accuracy | Macro F1 | Tham số |
|---|---:|---:|---:|
| M1 CNN | 63.69% | 62.30% | 1,764,519 |
| M2 ResNet18 Transfer Learning | 66.30% | 64.35% | 11,180,103 |

Các metric này lấy từ `results/<model>/metrics.json`. Trường `inference_fps_batch_estimate` là throughput ước tính khi suy luận theo batch trên thiết bị ghi trong metrics, không phải FPS end-to-end của camera trên CPU. FPS ứng dụng được hiển thị khi chạy app và phụ thuộc phần cứng, detector, kích thước khung hình và số khuôn mặt.

### Huấn luyện lại khi thực sự cần

```bash
python src/train.py --data data/fer2013 --model cnn --epochs 20 --batch-size 64
python src/train.py --data data/fer2013 --model transfer --epochs 10 --batch-size 32
```

Notebook `notebooks/train.ipynb` giữ quy trình huấn luyện và đánh giá. Tập train được chia stratified 90%/10% thành train/validation; tập test giữ riêng để đánh giá cuối cùng. Hàm mất mát dùng class weights để giảm ảnh hưởng mất cân bằng lớp. Lần train Transfer Learning đầu tiên có thể cần internet để tải trọng số ResNet18 ImageNet.

## 4. Chạy ứng dụng

### Lệnh bắt buộc theo đề

```bash
python src/app.py --source http://<ip>:8080/video --model cnn
python src/app.py --source test.jpg --model transfer
```

Thay `<ip>` bằng địa chỉ IP thực tế hiển thị trong ứng dụng IP Webcam/DroidCam. Laptop và điện thoại phải cùng Wi-Fi (hoặc laptop kết nối hotspot điện thoại). Mở URL trong trình duyệt laptop trước để kiểm tra luồng.

### Các nguồn khác và tuỳ chọn

```bash
python src/app.py --source 0 --model cnn
python src/app.py --source sample.mp4 --model transfer
python src/app.py --source face.jpg --model transfer --confidence 0.5
python src/app.py --source face.jpg --model cnn --no-display
```

`--source` nhận URL IP camera, chỉ số webcam, ảnh hoặc video. `--model` chọn `cnn` hoặc `transfer`; `--confidence` đặt ngưỡng tin cậy. Khi đầu vào là ảnh nhỏ (ví dụ ảnh FER-2013 48×48 đã crop sẵn), nếu Haar Cascade không tìm thấy mặt thì ứng dụng dùng toàn ảnh làm vùng mặt. Với ảnh/video thông thường, app dùng Haar Cascade để phát hiện nhiều khuôn mặt.

Ứng dụng hiển thị bounding box, nhãn cảm xúc, confidence và FPS. Nếu luồng IP mất kết nối, app thông báo và thử kết nối lại tối đa 5 lần mặc định rồi thoát sạch; điều chỉnh bằng `--max-reconnect-attempts`. `--no-display` hữu ích trên máy chủ/Colab không có GUI. Với ảnh đầu vào, ảnh kết quả được lưu vào `results/demo_<model>.jpg`.

## 5. Kết quả và báo cáo

- `results/<model>/metrics.json`: Accuracy, Macro F1, tham số, confusion matrix và throughput batch.
- `classification_report.txt`: precision/recall/F1 theo lớp.
- `training_curves.png`: train/validation loss và accuracy.
- `confusion_matrix.png`: confusion matrix trên test.
- `misclassified_examples.csv` và `misclassified_grid.jpg`: ví dụ dự đoán sai để phân tích.
- `report.pdf`: báo cáo 5 trang, gồm dữ liệu, mô hình, so sánh, confusion matrix, đồ thị và phân tích lỗi.

## 6. Lưu ý khi demo

1. Kiểm tra camera và URL trước buổi đánh giá; IP có thể thay đổi khi đổi mạng.
2. Chạy lệnh từ thư mục gốc project để đường dẫn `weights/`, `results/` và `config.yaml` được phân giải đúng.
3. FPS đo trong app là FPS xử lý vòng lặp end-to-end; phải kiểm tra trên laptop thực tế để xác nhận mục tiêu 10 FPS.
4. Mô hình dự đoán biểu cảm trên ảnh, không xác định chắc chắn trạng thái cảm xúc thật của một người.
