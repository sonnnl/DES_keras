# Ứng Dụng Học Máy Để Nâng Cao Hiệu Suất Mã Hóa DES

Dự án này sử dụng học máy để cải thiện hiệu suất của thuật toán mã hóa DES (Data Encryption Standard) trong các hệ thống ngân hàng số. Phương pháp chính được triển khai là **dự đoán khóa mã hóa** bằng mạng neural.

## Phiên Bản Cải Tiến

Phiên bản cải tiến này tập trung vào:

- Tăng lượng dữ liệu huấn luyện (100,000 mẫu)
- Mô hình mạng neural sâu hơn và phức tạp hơn
- Xuất dữ liệu huấn luyện và kết quả dự đoán ra CSV
- Giao diện mô phỏng ngân hàng số với khả năng xuất dữ liệu

## Cấu Trúc Dự Án

- `des_key_predictor_improved.py`: Mô hình học máy cải tiến
- `trainer.py`: Script huấn luyện mô hình với nhiều tham số tùy chỉnh
- `bank_digital_demo_improved.py`: Ứng dụng mô phỏng ngân hàng số
- `export_simple_data_csv.py`: Tool xuất dữ liệu đơn giản ra CSV

## Cài Đặt

1. Cài đặt các thư viện cần thiết:

```
pip install -r requirements.txt
```

2. Huấn luyện mô hình:

```
python trainer.py
```

3. Chạy ứng dụng mô phỏng:

```
python bank_digital_demo_improved.py
```

## Tính Năng

### 1. Huấn Luyện Mô Hình Cải Tiến

- Dữ liệu huấn luyện tăng lên 100,000 mẫu
- Kiến trúc mạng neural sâu hơn với nhiều lớp
- Cơ chế Early Stopping và Model Checkpoint
- Xuất dữ liệu huấn luyện ra CSV để phân tích

### 2. Ứng Dụng Mô Phỏng Ngân Hàng

- Mô phỏng giao dịch ngân hàng sử dụng mã hóa DES
- So sánh hiệu suất giữa DES truyền thống và DES hỗ trợ ML
- Biểu đồ độ chính xác dự đoán bit và thời gian xử lý
- Xuất dữ liệu giao dịch ra CSV để phân tích

## Kết Quả và Phân Tích

Mô hình sau khi huấn luyện với 100,000 mẫu có thể đạt được độ chính xác dự đoán bit lên đến 60-70%, cải thiện đáng kể so với phiên bản trước. Tuy nhiên, việc dự đoán hoàn toàn chính xác khóa DES vẫn là một thách thức lớn do cấu trúc của thuật toán.

Dự án minh họa tiềm năng của học máy trong việc tối ưu hóa các hệ thống mã hóa, nhưng cũng cho thấy giới hạn của phương pháp này khi áp dụng với các thuật toán mã hóa được thiết kế để chống lại phân tích.

## Hướng Phát Triển

- Áp dụng các kỹ thuật feature engineering đặc thù dựa trên cấu trúc DES
- Thử nghiệm các kiến trúc mạng neural khác (CNN, RNN)
- Tối ưu hóa hiệu suất mô hình trên các phần cứng khác nhau

## Ý Tưởng Chính

Khi mã hóa và giải mã dữ liệu bằng DES trong hệ thống ngân hàng số, việc xử lý khóa có thể tạo ra nút thắt cổ chai về hiệu suất. Dự án này sử dụng mạng neural để dự đoán khóa dựa trên cặp plaintext-ciphertext, giúp tăng tốc quá trình xử lý mã hóa.

## Tính Năng

- Tạo dữ liệu huấn luyện từ các cặp plaintext, ciphertext và khóa
- Xây dựng và huấn luyện mô hình mạng neural để dự đoán khóa DES
- Đo lường và so sánh hiệu suất giữa phương pháp truyền thống và phương pháp sử dụng học máy
- Lưu và tải mô hình đã huấn luyện để sử dụng trong tương lai

## Hiệu Quả

Mô hình học máy có thể dự đoán khóa DES với độ chính xác cao, giúp cải thiện hiệu suất mã hóa/giải mã đáng kể trong các hệ thống ngân hàng số. Mô hình đặc biệt hiệu quả trong các môi trường yêu cầu tốc độ xử lý cao như giao dịch tài chính trực tuyến.

## Lưu Ý An Ninh

Đây là một dự án học thuật và thí nghiệm. Trong môi trường thực tế, việc sử dụng học máy để dự đoán khóa mã hóa cần được cân nhắc cẩn thận về mặt an ninh. Các hệ thống ngân hàng số nên sử dụng các tiêu chuẩn mã hóa mạnh hơn (AES) và các biện pháp bảo mật nhiều lớp.

## Ứng Dụng Thực Tế Trong Ngân Hàng Số

- **Xử lý giao dịch tốc độ cao**: Rút ngắn thời gian mã hóa/giải mã trong các giao dịch tài chính
- **Hệ thống xác thực**: Tối ưu hóa các quy trình xác thực người dùng
- **Bảo mật dữ liệu**: Cải thiện hiệu suất mã hóa dữ liệu khách hàng
- **Mobile banking**: Giảm độ trễ trong các ứng dụng ngân hàng di động
