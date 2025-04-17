#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
DES Key Predictor - Mô hình dự đoán khóa DES dựa trên plaintext và ciphertext
"""

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
import os
import time
import random
from Crypto.Cipher import DES
    import joblib

class DESKeyPredictor:
    """
    Lớp dự đoán khóa DES sử dụng deep learning
    """
    
    def __init__(self):
        """Khởi tạo DESKeyPredictor"""
        self.model = None
        self.feature_scaler = None
        self.loaded = False
        
    def load_model(self, path=None):
        """
        Tải mô hình đã huấn luyện sẵn
        
        Args:
            path: Đường dẫn đến mô hình (mặc định: tìm trong thư mục 'models')
        
        Returns:
            self: instance hiện tại
        """
        if path is None:
            # Tìm file model mới nhất trong thư mục 'models'
            model_dir = 'models'
            if not os.path.exists(model_dir):
                print(f"Thư mục {model_dir} không tồn tại")
                return self
                
            model_files = [f for f in os.listdir(model_dir) if f.startswith('des_key_predictor_model_') and f.endswith('.keras')]
            if not model_files:
                print(f"Không tìm thấy mô hình trong {model_dir}")
                return self
                
            # Sắp xếp theo thời gian tạo giảm dần để lấy file mới nhất
            model_files.sort(reverse=True)
            path = os.path.join(model_dir, model_files[0])
            
        try:
            print(f"Đang tải mô hình từ {path}")
            self.model = models.load_model(path)
            
            # Tải feature scaler nếu có
            scaler_path = path.replace('.keras', '_scaler.joblib')
            if os.path.exists(scaler_path):
                self.feature_scaler = joblib.load(scaler_path)
                print(f"Đã tải feature scaler từ {scaler_path}")
            
            self.loaded = True
            print("Đã tải mô hình thành công")
                return self
            except Exception as e:
            print(f"Lỗi khi tải mô hình: {e}")
            return self
    
    def predict_key(self, plaintext, ciphertext):
        """
        Dự đoán khóa DES từ cặp plaintext/ciphertext
        
        Args:
            plaintext: Plaintext (bytes)
            ciphertext: Ciphertext (bytes)
            
        Returns:
            bytes: Khóa DES dự đoán (8 bytes)
        """
        if not self.loaded or self.model is None:
            # Nếu mô hình chưa được tải, trả về khóa ngẫu nhiên
            print("Mô hình chưa được tải, trả về khóa ngẫu nhiên")
            return os.urandom(8)
            
        # Trích xuất đặc trưng từ plaintext và ciphertext
        features = self._prepare_features(plaintext, ciphertext)
        
        # Chuẩn hóa đặc trưng nếu có scaler
        if self.feature_scaler is not None:
            features = self.feature_scaler.transform([features])[0]
        
        # Dự đoán
        try:
                    prediction = self.model.predict(np.array([features]), verbose=0)[0]
            
            # Chuyển đổi dự đoán thành bytes
            key_bytes = bytearray(8)
            for i in range(8):
                byte_val = 0
                for j in range(8):
                    bit_idx = i * 8 + j
                    if bit_idx < len(prediction) and prediction[bit_idx] >= 0.5:
                        byte_val |= (1 << (7 - j))
                key_bytes[i] = byte_val
                
            return bytes(key_bytes)
        except Exception as e:
            print(f"Lỗi khi dự đoán: {e}")
            return os.urandom(8)
            
    def _prepare_features(self, plaintext, ciphertext):
        """
        Trích xuất đặc trưng từ plaintext và ciphertext
        
        Args:
            plaintext: Plaintext (bytes)
            ciphertext: Ciphertext (bytes)
            
        Returns:
            numpy.ndarray: Vector đặc trưng
        """
        # Convert plaintext and ciphertext to bit arrays
        plaintext_bits = []
        for b in plaintext[:8]:  # Chỉ xử lý 8 bytes đầu tiên
            for i in range(7, -1, -1):
                plaintext_bits.append((b >> i) & 1)
                
        ciphertext_bits = []
        for b in ciphertext[:8]:  # Chỉ xử lý 8 bytes đầu tiên
            for i in range(7, -1, -1):
                ciphertext_bits.append((b >> i) & 1)
                
        # Kết hợp các đặc trưng
        features = plaintext_bits + ciphertext_bits
        
        # Đảm bảo độ dài cố định
        if len(features) < 128:
            features += [0] * (128 - len(features))
        elif len(features) > 128:
            features = features[:128]
            
        return np.array(features, dtype=np.float32)

    def build_model(self):
        """
        Xây dựng mô hình neural network
        
        Returns:
            tensorflow.keras.models.Model: Mô hình được xây dựng
        """
        input_layer = layers.Input(shape=(128,))
        
        # Các lớp ẩn
        x = layers.Dense(256, activation='relu')(input_layer)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.2)(x)
        
        x = layers.Dense(256, activation='relu')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.2)(x)
        
        x = layers.Dense(128, activation='relu')(x)
        x = layers.BatchNormalization()(x)
        
        # Lớp đầu ra (64 bits cho khóa DES)
        output_layer = layers.Dense(64, activation='sigmoid')(x)
        
        model = models.Model(inputs=input_layer, outputs=output_layer)
        model.compile(
            optimizer='adam',
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        
        return model

def main():
    """
    Hàm main để test
    """
    # Khởi tạo predictor
    predictor = DESKeyPredictor()
    
    # Tải mô hình nếu có
    predictor.load_model()
    
    # Test dự đoán
    plaintext = b"TESTDATA"
            key = os.urandom(8)
    
    # Mã hóa
        cipher = DES.new(key, DES.MODE_ECB)
    ciphertext = cipher.encrypt(plaintext)
    
    print(f"Plaintext: {plaintext}")
    print(f"Ciphertext: {ciphertext.hex()}")
    print(f"Actual key: {key.hex()}")
    
    # Dự đoán
    predicted_key = predictor.predict_key(plaintext, ciphertext)
    print(f"Predicted key: {predicted_key.hex()}")
        
        # Tính độ chính xác bit
    correct_bits = 0
    total_bits = 64
    
        for i in range(8):
        for j in range(8):
            if ((key[i] >> j) & 1) == ((predicted_key[i] >> j) & 1):
                correct_bits += 1
                
    accuracy = correct_bits / total_bits * 100
    print(f"Bit-level accuracy: {accuracy:.2f}%")

if __name__ == "__main__":
    main() 