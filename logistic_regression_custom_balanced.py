
import numpy as np

class LogisticRegressionCustomBalanced():
    def __init__(self, learning_rate=0.01, epochs=1000, random_state=0):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.random_state = random_state
        self.weights = None
        self.bias = None
        self.weights_class = None

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    '''
    Nhân trọng số lớp với cross-entropy loss cho từng mẫu. 
    Điều này làm tăng tầm quan trọng của các lớp thiểu số trong quá trình huấn luyện.
    '''
    def compute_loss(self, y_true, y_pred):
        loss = -np.mean(self.weights_class[y_true.astype(int)] * 
                        (y_true * np.log(y_pred + 1e-9) + 
                        (1 - y_true) * np.log(1 - y_pred + 1e-9)))
        return loss

    '''
    Gradient của hàm mất mát được điều chỉnh bằng trọng số lớp, 
    giúp cập nhật trọng số mô hình một cách phù hợp với tầm quan trọng của từng lớp.
    '''
    def compute_gradient(self, x, y_true, y_pred):
        errors = y_pred - y_true
        weighted_errors = errors * self.weights_class[y_true.astype(int)]
        dw = np.dot(x.T, weighted_errors) / x.shape[0]
        db = np.mean(weighted_errors)
        return dw, db

    def update_parameters(self, weights, bias, dw, db):
        weights -= self.learning_rate * dw
        bias -= self.learning_rate * db
        return weights, bias

    def fit(self, x_train, y_train):
        np.random.seed(self.random_state)
        self.weights = np.random.randn(x_train.shape[1])
        self.bias = 0
        
        # Tính trọng số cho mỗi lớp
        classes, counts = np.unique(y_train, return_counts=True) # Tìm các lớp duy nhất và đếm số lượng mẫu trong mỗi lớp.
        total = len(y_train)
        self.weights_class = {}
        for cls, count in zip(classes, counts):
            self.weights_class[cls] = total / (len(classes) * count) # Tính trọng số cho mỗi lớp bằng cách lấy tổng số mẫu chia cho (số lớp × số mẫu trong lớp hiện tại).
        
        # Tạo một mảng trọng số tương ứng với từng mẫu trong dữ liệu huấn luyện.
        self.weights_class = np.array([self.weights_class[cls] for cls in y_train])

        for epoch in range(self.epochs):
            # Dự đoán
            linear_model = np.dot(x_train, self.weights) + self.bias
            y_pred = self.sigmoid(linear_model)

            # Tính toán hàm loss
            loss = self.compute_loss(y_train, y_pred)

            # Tính gradient
            dw, db = self.compute_gradient(x_train, y_train, y_pred)

            # Cập nhật trọng số và bias
            self.weights, self.bias = self.update_parameters(self.weights, self.bias, dw, db)

            # In loss mỗi 100 epochs
            if (epoch+1) % 100 == 0:
                print(f'Epoch {epoch+1}/{self.epochs} - Loss: {loss:.4f}')

    def predict_proba(self, x):
        linear_model = np.dot(x, self.weights) + self.bias
        return self.sigmoid(linear_model)

    def predict(self, x, threshold=0.5):
        y_pred_prob = self.predict_proba(x)
        return (y_pred_prob >= threshold).astype(int)