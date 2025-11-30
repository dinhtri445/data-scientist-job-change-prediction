import numpy as np

class LogisticRegressionNumPy:
    def __init__(self, learning_rate=0.01, n_iters=1000):
        self.lr = learning_rate
        self.n_iters = n_iters
        self.weights = None
        self.bias = None
        self.costs = [] # Để vẽ biểu đồ loss

    def _sigmoid(self, z):
        # Hàm kích hoạt Sigmoid: 1 / (1 + e^-z)
        # np.clip để tránh lỗi tràn số (overflow) khi z quá lớn/nhỏ
        z = np.clip(z, -250, 250)
        return 1 / (1 + np.exp(-z))

    def _compute_loss(self, y_true, y_pred):
        # Hàm mất mát Log Loss 
        # J = - (1/m) * sum(y*log(y_hat) + (1-y)*log(1-y_hat))
        m = len(y_true)
      
        epsilon = 1e-15
        y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
        
        loss = - (1/m) * np.sum(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
        return loss

    def fit(self, X, y):
        """
        Huấn luyện mô hình dùng Gradient Descent.
        """
        n_samples, n_features = X.shape
        
        
        self.weights = np.zeros(n_features)
        self.bias = 0

        # Vòng lặp tối ưu (Gradient Descent)
        for i in range(self.n_iters):
            # A. Tính toán dự đoán (Forward Pass)
            # z = w*x + b
            linear_model = np.dot(X, self.weights) + self.bias
            # y_hat = sigmoid(z)
            y_predicted = self._sigmoid(linear_model)

            # B. Tính Gradient (Đạo hàm)
            # dw = (1/m) * X.T * (y_hat - y)
            dw = (1 / n_samples) * np.dot(X.T, (y_predicted - y))
            # db = (1/m) * sum(y_hat - y)
            db = (1 / n_samples) * np.sum(y_predicted - y)

            # C. Cập nhật tham số (Update Parameters)
            self.weights -= self.lr * dw
            self.bias -= self.lr * db
            
            # (Optional) Lưu loss để theo dõi
            if i % 100 == 0:
                cost = self._compute_loss(y, y_predicted)
                self.costs.append(cost)
                
    def predict_proba(self, X):
        """Trả về xác suất (0.0 - 1.0)"""
        linear_model = np.dot(X, self.weights) + self.bias
        return self._sigmoid(linear_model)

    def predict(self, X, threshold=0.5):
        """Trả về nhãn (0 hoặc 1)"""
        y_predicted_cls = self.predict_proba(X)
        return [1 if i > threshold else 0 for i in y_predicted_cls]

# --- HÀM ĐÁNH GIÁ (EVALUATION METRICS) ---
def accuracy_score_numpy(y_true, y_pred):
    accuracy = np.sum(y_true == y_pred) / len(y_true)
    return accuracy

def confusion_matrix_numpy(y_true, y_pred):
    # Tính TP, TN, FP, FN
    # y_true và y_pred phải là mảng numpy
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    TP = np.sum((y_true == 1) & (y_pred == 1))
    TN = np.sum((y_true == 0) & (y_pred == 0))
    FP = np.sum((y_true == 0) & (y_pred == 1))
    FN = np.sum((y_true == 1) & (y_pred == 0))
    
    return np.array([[TN, FP], [FN, TP]])

def classification_report_numpy(y_true, y_pred):
    cm = confusion_matrix_numpy(y_true, y_pred)
    TN, FP, FN, TP = cm.ravel()
    
    accuracy = (TP + TN) / (TP + TN + FP + FN)
    
    # Precision = TP / (TP + FP)
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0
    
    # Recall = TP / (TP + FN)
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0
    
    # F1 = 2 * (Pre * Rec) / (Pre + Rec)
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    print(f"Accuracy:  {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1-Score:  {f1:.4f}")
    
    return {"accuracy": accuracy, "f1": f1, "recall": recall}