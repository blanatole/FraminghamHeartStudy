# Nghiên cứu tim Framingham

### Nội dung

#### 1. **Tải dữ liệu**
    - Nhập các gói thư viện
    - Đọc dữ liệu
    - Kiểm tra kích thước dữ liệu
    - Kiểm tra kiểu dữ liệu

#### 2. **Làm sạch dữ liệu & Phân tích khám phá dữ liệu (EDA)**
    - Kiểm tra giá trị Null
    - Bản đồ nhiệt tương quan
    - Điền giá trị Null
    - Thống kê mô tả
    - EDA (Gia đoạn 1)
    - Mã hóa trung bình
    - EDA (Gia đoạn 2)
    - Chuyển đổi logarith

#### 3. **Chuẩn hóa**
    - StandardScaler

#### 4. **Mô hình hóa**
    - Chia dữ liệu & Chọn thuật toán
    - Triển khai Hồi quy Logistic

#### 5. **Cải tiến mô hình**

#### 6. **Xây dựng ứng dụng**

### Chi tiết :

Bộ dữ liệu là một tập con khá nhỏ của bộ dữ liệu FHS, có 4240 quan sát và 16 biến. Các biến như sau:

- sex : giới tính của các quan sát. Biến là nhị phân được đặt tên là “male” trong bộ dữ liệu. (0 = "Nữ")
- age : Tuổi tại thời điểm khám sức khỏe bằng năm.
- education : Một biến phân loại về trình độ học vấn của người tham gia, với các mức: Một ít trung học phổ thông (1), trung học phổ thông/GED (2), một ít học cao đẳng/trường nghề (3), đại học (4)
- currentSmoker: Hút thuốc lá hiện tại tại thời điểm khám
- cigsPerDay: Số lượng điếu thuốc lá hàng ngày
- BPmeds: Sử dụng thuốc hạ huyết áp tại cuộc khám
- prevalentStroke: Đột quỵ phổ biến (0 = không mắc bệnh)
- prevalentHyp: Tăng huyết áp phổ biến. Người tham gia được xác định là bị tăng huyết áp nếu đang điều trị
- diabetes: Bệnh tiểu đường theo tiêu chí của cuộc khám đầu tiên được điều trị
- totChol: Tổng cholesterol (mg/dL)
- sysBP: Huyết áp tâm thu (mmHg)
- diaBP: Huyết áp tâm trương (mmHg)
- BMI: Chỉ số khối cơ thể, cân nặng (kg)/chiều cao (m)^2
- heartRate: Nhịp tim (nhịp/phút)
- glucose: Mức đường huyết (mg/dL)

Và cuối cùng là biến phản hồi: CHD: Bệnh tim mạch vành.


**Bảng câu hỏi :**

- Mỗi thuộc tính trong bộ dữ liệu có sự phân bố như thế nào?

- Chúng ta có thể lấy số lượng CHD theo giới tính không?

- Chúng ta có thể nhóm người ở độ tuổi nhất định lại và tìm hiểu huyết áp tâm thu và huyết áp tâm trương ảnh hưởng như thế nào theo nhóm tuổi?

- Biến mục tiêu của chúng ta được phân bố như thế nào? Nó có bị mất cân bằng không?


**Mục tiêu :**

- Mục tiêu là xây dựng một số mô hình dự đoán trên bộ dữ liệu FHS và xem xét một số kỹ thuật khám phá và mô hình hóa.

### Hướng dẫn chạy ứng dụng

#### 1. Thêm một terminal mới.

#### 2. Cài đặt các thư viện cần thiết theo cấu trúc: 
``` bash
pip install <library-name>
```

#### 3. Chạy lệnh: 
```bash 
streamlit run app.py
```
#### 4. Hoàn tất
