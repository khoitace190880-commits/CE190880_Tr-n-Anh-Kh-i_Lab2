# Lab 2 — Nhận diện cảm xúc khuôn mặt bằng Deep Learning

> Báo cáo mẫu để hoàn thiện sau khi chạy huấn luyện. Thay mọi trường `[điền kết quả]` bằng số liệu thực tế từ `results/cnn/metrics.json` và `results/transfer/metrics.json`.

## 1. Dữ liệu và tiền xử lý
FER-2013 trong gói được cung cấp có 28.709 ảnh train và 7.178 ảnh test, kích thước ảnh 48×48 grayscale, 7 lớp. Train được chia phân tầng thành train/validation theo tỷ lệ 90/10; tập test giữ riêng để đánh giá cuối cùng. Phân bố lớp train: angry 3.995; disgust 436; fear 4.097; happy 7.215; neutral 4.965; sad 4.830; surprise 3.171. Mất cân bằng được xử lý bằng class-weighted cross entropy và augmentation nhẹ (lật ngang, xoay nhỏ) trên train.

## 2. M1 — CNN tự thiết kế
Kiến trúc gồm bốn khối Conv2D–BatchNorm–ReLU–MaxPool với số kênh 32, 64, 128, 256; sau đó Flatten, fully connected 256, Dropout 0.45 và đầu ra 7 lớp. Kích thước đầu vào 1×48×48. Optimizer AdamW, learning rate mặc định 0.001, weight decay 0.0001. Số epoch thực tế: [điền]. Kết quả test accuracy: [điền]; macro F1: [điền].

## 3. M2 — Transfer learning
Sử dụng ResNet18 pretrained trên ImageNet, thay lớp fully-connected bằng đầu ra 7 lớp. Ảnh grayscale được chuyển thành 3 kênh, resize 224×224 và chuẩn hóa theo ImageNet. Optimizer AdamW, learning rate mặc định 0.0001. Số epoch thực tế: [điền]. Kết quả test accuracy: [điền]; macro F1: [điền].

## 4. So sánh M1 và M2
| Chỉ số | M1 CNN | M2 ResNet18 |
|---|---:|---:|
| Test accuracy | [điền] | [điền] |
| Macro F1-score | [điền] | [điền] |
| Số tham số | [điền] | [điền] |
| Inference FPS | [điền] | [điền] |
| Đạt ngưỡng đề (60% / 65%) | [có/không] | [có/không] |

## 5. Phân tích lỗi
Chèn confusion matrix của cả hai mô hình. Chọn tối thiểu 10 ảnh dự đoán sai từ `results/<model>/misclassified_grid.jpg` hoặc đối chiếu file CSV và ghi nhận xét cho từng ảnh: nhãn thật, nhãn dự đoán, confidence, nguyên nhân có thể (biểu cảm gần nhau, góc mặt, ánh sáng, che khuất, chất lượng ảnh). Chỉ ghi nhận xét phù hợp với ảnh quan sát được.

## 6. Ứng dụng realtime
Sử dụng OpenCV Haar Cascade để phát hiện khuôn mặt và mô hình đã huấn luyện để phân loại từng khuôn mặt. Ứng dụng hiển thị bounding box, nhãn, confidence và FPS; hỗ trợ IP camera, ảnh tĩnh, video file và webcam index. Ảnh chụp demo: [chèn ảnh]. FPS đo trên laptop CPU: [điền].

## 7. Kết luận
Nêu mô hình có kết quả tốt hơn theo accuracy/macro F1, sự đánh đổi giữa tốc độ và số tham số, các lớp dễ nhầm lẫn và hướng cải thiện. Không khẳng định đạt yêu cầu nếu số liệu thực tế chưa đáp ứng ngưỡng.
