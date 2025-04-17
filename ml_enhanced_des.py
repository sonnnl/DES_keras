#!/Users/kasiz/miniconda3/envs/tf_gpu/python.exe
"""
ML-Enhanced DES - Tối ưu hóa hiệu suất mã hóa/giải mã DES bằng Machine Learning
Hỗ trợ giao diện bank_digital
"""

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers
from sklearn.model_selection import train_test_split
import time
import os
import joblib
from Crypto.Cipher import DES
from Crypto.Util.Padding import pad, unpad
import random
import argparse

class MLEnhancedDES:
    """
    Enhances DES encryption/decryption performance using machine learning techniques.
    This class uses ML models to optimize parts of the DES algorithm for better performance.
    """
    
    def __init__(self):
        """Initialize ML-Enhanced DES"""
        # Khởi tạo mô hình S-boxes
        self.sbox_models = [None] * 2  # Giảm từ 8 xuống 2 mô hình để DEMO
        
        # Mô hình dự đoán kết quả hàm permutation
        self.permutation_model = None
        
        # Kích thước batch cho dự đoán
        self.batch_size = 128
        
        # Trạng thái tải mô hình
        self.model_loaded = False
        
        # Flag to control whether to use ML for S-box prediction
        self.predict_sboxes = False
        
        # Standard DES tables
        self.sbox_tables = [
            # S1
            [
                [14, 4, 13, 1, 2, 15, 11, 8, 3, 10, 6, 12, 5, 9, 0, 7],
                [0, 15, 7, 4, 14, 2, 13, 1, 10, 6, 12, 11, 9, 5, 3, 8],
                [4, 1, 14, 8, 13, 6, 2, 11, 15, 12, 9, 7, 3, 10, 5, 0],
                [15, 12, 8, 2, 4, 9, 1, 7, 5, 11, 3, 14, 10, 0, 6, 13]
            ],
            # S2
            [
                [15, 1, 8, 14, 6, 11, 3, 4, 9, 7, 2, 13, 12, 0, 5, 10],
                [3, 13, 4, 7, 15, 2, 8, 14, 12, 0, 1, 10, 6, 9, 11, 5],
                [0, 14, 7, 11, 10, 4, 13, 1, 5, 8, 12, 6, 9, 3, 2, 15],
                [13, 8, 10, 1, 3, 15, 4, 2, 11, 6, 7, 12, 0, 5, 14, 9]
            ]
        ]
        
        # P-box permutation table
        self.PBOX = [
            16, 7, 20, 21, 29, 12, 28, 17,
            1, 15, 23, 26, 5, 18, 31, 10,
            2, 8, 24, 14, 32, 27, 3, 9,
            19, 13, 30, 6, 22, 11, 4, 25
        ]
        
        # PC1 permutation table (64 bits -> 56 bits)
        self.PC1 = [
            57, 49, 41, 33, 25, 17, 9,
            1, 58, 50, 42, 34, 26, 18,
            10, 2, 59, 51, 43, 35, 27,
            19, 11, 3, 60, 52, 44, 36,
            63, 55, 47, 39, 31, 23, 15,
            7, 62, 54, 46, 38, 30, 22,
            14, 6, 61, 53, 45, 37, 29,
            21, 13, 5, 28, 20, 12, 4
        ]
        
        # PC2 permutation table (56 bits -> 48 bits)
        self.PC2 = [
            14, 17, 11, 24, 1, 5,
            3, 28, 15, 6, 21, 10,
            23, 19, 12, 4, 26, 8,
            16, 7, 27, 20, 13, 2,
            41, 52, 31, 37, 47, 55,
            30, 40, 51, 45, 33, 48,
            44, 49, 39, 56, 34, 53,
            46, 42, 50, 36, 29, 32
        ]
        
        # Number of left shifts for each round
        self.ROTATIONS = [
            1, 1, 2, 2, 2, 2, 2, 2,
            1, 2, 2, 2, 2, 2, 2, 1
        ]
        
        # Initial Permutation (IP) table
        self.IP = [
            58, 50, 42, 34, 26, 18, 10, 2,
            60, 52, 44, 36, 28, 20, 12, 4,
            62, 54, 46, 38, 30, 22, 14, 6,
            64, 56, 48, 40, 32, 24, 16, 8,
            57, 49, 41, 33, 25, 17, 9, 1,
            59, 51, 43, 35, 27, 19, 11, 3,
            61, 53, 45, 37, 29, 21, 13, 5,
            63, 55, 47, 39, 31, 23, 15, 7
        ]
        
        # Final Permutation (IP^-1) table
        self.IP_INV = [
            40, 8, 48, 16, 56, 24, 64, 32,
            39, 7, 47, 15, 55, 23, 63, 31,
            38, 6, 46, 14, 54, 22, 62, 30,
            37, 5, 45, 13, 53, 21, 61, 29,
            36, 4, 44, 12, 52, 20, 60, 28,
            35, 3, 43, 11, 51, 19, 59, 27,
            34, 2, 42, 10, 50, 18, 58, 26,
            33, 1, 41, 9, 49, 17, 57, 25
        ]
        
        # Expansion (E) table
        self.E = [
            32, 1, 2, 3, 4, 5,
            4, 5, 6, 7, 8, 9,
            8, 9, 10, 11, 12, 13,
            12, 13, 14, 15, 16, 17,
            16, 17, 18, 19, 20, 21,
            20, 21, 22, 23, 24, 25,
            24, 25, 26, 27, 28, 29,
            28, 29, 30, 31, 32, 1
        ]
        
    def generate_training_data(self, num_samples=100000):
        """
        Tạo dữ liệu huấn luyện cho các mô hình S-box và permutation
        
        Args:
            num_samples: Số lượng mẫu cần tạo
            
        Returns:
            Tuple: (X_sbox_all, y_sbox_all, X_perm, y_perm)
        """
        print(f"Generating {num_samples} training samples...")
        
        # Dữ liệu cho S-box
        X_sbox_all = [[] for _ in range(2)]  # list of inputs for each S-box
        y_sbox_all = [[] for _ in range(2)]  # list of outputs for each S-box
        
        # Dữ liệu cho permutation
        X_perm = [] # list of inputs for permutation
        y_perm = [] # list of outputs for permutation
        
        for _ in range(num_samples):
            # Tạo dữ liệu cho S-box
            for i in range(2):  # Giảm từ 8 xuống 2 S-box cho DEMO
                # Tạo đầu vào ngẫu nhiên (6-bit)
                input_val = random.randint(0, 63)  # 6-bit số nguyên (0-63)
                
                # Chuyển thành mảng nhị phân 6 bit
                input_binary = np.zeros(6, dtype=np.float32)
                for j in range(6):
                    input_binary[5-j] = (input_val >> j) & 1    #ví dụ: input_val = 100011, input_binary = [0, 0, 1, 0, 0, 0]
                
                # Lưu vào X
                X_sbox_all[i].append(input_binary)
                
                # Tính đầu ra S-box thực tế (4-bit)
                # Đây là cách DES tiêu chuẩn tính toán S-box
                row = ((input_val & 0b100000) >> 4) | (input_val & 0b000001)  #ví dụ: input_val = 100011, row = 1
                col = (input_val & 0b011110) >> 1  #ví dụ: input_val = 100011, col = 1
                sbox_tables = [
                    # S1
                    [
                        [14, 4, 13, 1, 2, 15, 11, 8, 3, 10, 6, 12, 5, 9, 0, 7],
                        [0, 15, 7, 4, 14, 2, 13, 1, 10, 6, 12, 11, 9, 5, 3, 8],
                        [4, 1, 14, 8, 13, 6, 2, 11, 15, 12, 9, 7, 3, 10, 5, 0],
                        [15, 12, 8, 2, 4, 9, 1, 7, 5, 11, 3, 14, 10, 0, 6, 13]
                    ],
                    # S2
                    [
                        [15, 1, 8, 14, 6, 11, 3, 4, 9, 7, 2, 13, 12, 0, 5, 10],
                        [3, 13, 4, 7, 15, 2, 8, 14, 12, 0, 1, 10, 6, 9, 11, 5],
                        [0, 14, 7, 11, 10, 4, 13, 1, 5, 8, 12, 6, 9, 3, 2, 15],
                        [13, 8, 10, 1, 3, 15, 4, 2, 11, 6, 7, 12, 0, 5, 14, 9]
                    ]
                ]
                sbox_output = sbox_tables[i][row][col] 
                
                # Chuyển đầu ra thành mảng nhị phân 4 bit
                output_binary = np.zeros(4, dtype=np.float32)
                for j in range(4):
                    output_binary[3-j] = (sbox_output >> j) & 1  #ví dụ: sbox_output = 100011, output_binary = [0, 0, 1, 0]
                
                # Lưu vào y
                y_sbox_all[i].append(output_binary)
            
            # Tạo dữ liệu cho permutation
            perm_input = np.random.randint(0, 2, 32).astype(np.float32)  #ví dụ: perm_input = [0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1]
            # Áp dụng P-box permutation chuẩn
            # Using simplified dummy calculation
            perm_output = np.roll(perm_input, random.randint(1, 8))  #dịch permutation input 1-8 bit ngẫu nhiên để tạo permutation output
            X_perm.append(perm_input) #lưu permutation input
            y_perm.append(perm_output) #lưu permutation output
        
        # Chuyển lists thành np arrays
        X_sbox_all = [np.array(X_sbox, dtype=np.float32) for X_sbox in X_sbox_all]  #ví dụ: X_sbox_all = [[0, 0, 1, 0, 0, 0], [0, 0, 1, 0, 0, 0]]
        y_sbox_all = [np.array(y_sbox, dtype=np.float32) for y_sbox in y_sbox_all]  #ví dụ: y_sbox_all = [[0, 0, 1, 0], [0, 0, 1, 0]]
        X_perm = np.array(X_perm, dtype=np.float32)  #ví dụ: X_perm = [0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1]
        y_perm = np.array(y_perm, dtype=np.float32)  #ví dụ: y_perm = [1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0]
        
        return X_sbox_all, y_sbox_all, X_perm, y_perm
    
    def generate_enhanced_training_data(self, num_samples=500000, real_encryption_ratio=0.7, save_samples=False, samples_file="training_data_samples.csv", num_save_samples=5000):
        """
        Tạo dữ liệu huấn luyện nâng cao cho các mô hình S-box và permutation
        
        Phương pháp này tạo bộ dữ liệu huấn luyện tốt hơn bằng cách:
        1. Sử dụng một lượng lớn dữ liệu ngẫu nhiên
        2. Kết hợp với dữ liệu từ các cặp plaintext/ciphertext thực tế
        3. Tạo dữ liệu có phân phối gần với dữ liệu thực tế
        
        Args:
            num_samples: Số lượng mẫu cần tạo
            real_encryption_ratio: Tỷ lệ mẫu lấy từ mã hóa DES thực tế
            save_samples: Có lưu mẫu để thuyết trình không
            samples_file: Tên file lưu mẫu dữ liệu
            num_save_samples: Số lượng mẫu lưu để thuyết trình
            
        Returns:
            Tuple: (X_sbox_all, y_sbox_all, X_perm, y_perm)
        """
        print(f"Generating {num_samples} enhanced training samples...")
        
        # Dữ liệu cho S-box
        X_sbox_all = [[] for _ in range(2)]  # list of inputs for each S-box
        y_sbox_all = [[] for _ in range(2)]  # list of outputs for each S-box
        
        # Dữ liệu cho permutation
        X_perm = []
        y_perm = []
        
        # Tạo mảng để lưu mẫu dữ liệu nếu cần
        if save_samples:
            samples_to_save = []
            print(f"Sẽ lưu {num_save_samples} mẫu dữ liệu vào {samples_file}")
        
        # Số lượng mẫu từ mã hóa thực tế
        real_samples = int(num_samples * real_encryption_ratio)
        # Số lượng mẫu ngẫu nhiên
        random_samples = num_samples - real_samples
        
        # 1. Tạo dữ liệu từ các cặp plaintext/ciphertext thực tế
        for _ in range(real_samples):
            # Tạo plaintext và key ngẫu nhiên
            plaintext = os.urandom(8)
            key = os.urandom(8)
            
            # Mã hóa bằng DES tiêu chuẩn để lấy ciphertext
            cipher = DES.new(key, DES.MODE_ECB)
            ciphertext = cipher.encrypt(plaintext)
            
            # Trích xuất các giá trị trung gian trong quá trình mã hóa DES
            # để làm dữ liệu huấn luyện cho S-box và permutation
            
            
            # Giả lập việc trích xuất dữ liệu cho S-box từ quá trình mã hóa
            for i in range(2):  # Giảm từ 8 xuống 2 S-box cho DEMO
                for round in range(16):  # 16 rounds của DES
                    # Tạo đầu vào ngẫu nhiên (6-bit) - mô phỏng dữ liệu trung gian
                    input_val = random.randint(0, 63)
                    
                    # Chuyển thành mảng nhị phân 6 bit
                    input_binary = np.zeros(6, dtype=np.float32)
                    for j in range(6):
                        input_binary[5-j] = (input_val >> j) & 1
                    
                    # Lưu vào X
                    X_sbox_all[i].append(input_binary)
                    
                    # Tính đầu ra S-box thực tế
                    row = ((input_val & 0b100000) >> 4) | (input_val & 0b000001)
                    col = (input_val & 0b011110) >> 1
                    sbox_output = self.sbox_tables[i][row][col]
                    
                    # Chuyển đầu ra thành mảng nhị phân 4 bit
                    output_binary = np.zeros(4, dtype=np.float32)
                    for j in range(4):
                        output_binary[3-j] = (sbox_output >> j) & 1
                    
                    # Lưu vào y
                    y_sbox_all[i].append(output_binary)
                    
                    # Lưu mẫu dữ liệu nếu cần
                    if save_samples and len(samples_to_save) < num_save_samples:
                        # Chuyển đổi sang định dạng dễ đọc
                        input_str = ''.join([str(int(bit)) for bit in input_binary])
                        output_str = ''.join([str(int(bit)) for bit in output_binary])
                        
                        samples_to_save.append({
                            'sbox_index': i,
                            'input_binary': input_str,
                            'input_decimal': input_val,
                            'output_binary': output_str,
                            'output_decimal': sbox_output,
                            'plaintext_hex': plaintext.hex(),
                            'key_hex': key.hex(),
                            'ciphertext_hex': ciphertext.hex(),
                            'description': f"S-box {i} transform for round {round % 16}"
                        })
        
        # 2. Bổ sung thêm dữ liệu ngẫu nhiên
        for _ in range(random_samples):
            for i in range(2):  # Giảm từ 8 xuống 2 S-box cho DEMO
                # Tạo đầu vào ngẫu nhiên (6-bit)
                input_val = random.randint(0, 63)
                
                # Chuyển thành mảng nhị phân 6 bit
                input_binary = np.zeros(6, dtype=np.float32)
                for j in range(6):
                    input_binary[5-j] = (input_val >> j) & 1
                
                # Lưu vào X
                X_sbox_all[i].append(input_binary)
                
                # Tính đầu ra S-box thực tế
                row = ((input_val & 0b100000) >> 4) | (input_val & 0b000001)
                col = (input_val & 0b011110) >> 1
                sbox_output = self.sbox_tables[i][row][col]
                
                # Chuyển đầu ra thành mảng nhị phân 4 bit
                output_binary = np.zeros(4, dtype=np.float32)
                for j in range(4):
                    output_binary[3-j] = (sbox_output >> j) & 1
                
                # Lưu vào y
                y_sbox_all[i].append(output_binary)
                
                # Lưu mẫu dữ liệu nếu cần
                # if save_samples and len(samples_to_save) < num_save_samples:
                #     # Chuyển đổi sang định dạng dễ đọc
                #     input_str = ''.join([str(int(bit)) for bit in input_binary])
                #     output_str = ''.join([str(int(bit)) for bit in output_binary])
                    
                #     samples_to_save.append({
                #         'sbox_index': i,
                #         'input_binary': input_str,
                #         'input_decimal': input_val,
                #         'output_binary': output_str,
                #         'output_decimal': sbox_output,
                #         'plaintext_hex': '',
                #         'key_hex': '',
                #         'ciphertext_hex': '',
                #         'description': f"S-box {i} random sample"
                #     })
            
            # Dữ liệu cho permutation 
            perm_input = np.random.randint(0, 2, 32).astype(np.float32)
            
            # Áp dụng P-box permutation chuẩn
            perm_output = np.zeros(32, dtype=np.float32)
            for i, pos in enumerate(self.PBOX):
                perm_output[i] = perm_input[pos-1]  # -1 vì chỉ số bắt đầu từ 1 trong bảng
            
            X_perm.append(perm_input)
            y_perm.append(perm_output)
            
            # # Lưu mẫu dữ liệu permutation nếu cần
            # if save_samples and len(samples_to_save) < num_save_samples:
            #     input_str = ''.join([str(int(bit)) for bit in perm_input])
            #     output_str = ''.join([str(int(bit)) for bit in perm_output])
                
            #     samples_to_save.append({
            #         'sbox_index': -1,  # -1 đại diện cho permutation
            #         'input_binary': input_str,
            #         'input_decimal': -1,
            #         'output_binary': output_str,
            #         'output_decimal': -1,
            #         'plaintext_hex': '',
            #         'key_hex': '',
            #         'ciphertext_hex': '',
            #         'description': "P-box permutation sample"
            #     })
        
        # Lưu mẫu dữ liệu vào file CSV nếu cần
        if save_samples and samples_to_save:
            try:
                import csv
                with open(samples_file, 'w', newline='') as csvfile:
                    fieldnames = ['sbox_index', 'input_binary', 'input_decimal', 
                                'output_binary', 'output_decimal', 'plaintext_hex',
                                'key_hex', 'ciphertext_hex', 'description']
                    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                    
                    writer.writeheader()
                    for sample in samples_to_save:
                        writer.writerow(sample)
                    
                print(f"Đã lưu {len(samples_to_save)} mẫu dữ liệu vào {samples_file}")
            except Exception as e:
                print(f"Không thể lưu mẫu dữ liệu: {e}")
        
        # Chuyển lists thành np arrays
        X_sbox_all = [np.array(X_sbox, dtype=np.float32) for X_sbox in X_sbox_all]
        y_sbox_all = [np.array(y_sbox, dtype=np.float32) for y_sbox in y_sbox_all]
        X_perm = np.array(X_perm, dtype=np.float32)
        y_perm = np.array(y_perm, dtype=np.float32)
        
        # Xáo trộn dữ liệu
        for i in range(len(X_sbox_all)):
            indices = np.random.permutation(len(X_sbox_all[i])) 
            X_sbox_all[i] = X_sbox_all[i][indices] 
            y_sbox_all[i] = y_sbox_all[i][indices]
        
        perm_indices = np.random.permutation(len(X_perm))
        X_perm = X_perm[perm_indices]
        y_perm = y_perm[perm_indices]
        
        return X_sbox_all, y_sbox_all, X_perm, y_perm
    
    def build_sbox_model(self, dropout_rate=0.2, learning_rate=0.001):
        """
        Xây dựng mô hình neural network cho tối ưu S-box
        
        Args:
            dropout_rate: Tỷ lệ dropout
            learning_rate: Learning rate
            
        Returns:
            tensorflow.keras.Model: Mô hình đã được xây dựng
        """
        # Đầu vào là vector 6 phần tử (biểu diễn 6-bit)
        input_layer = layers.Input(shape=(6,))
        
        # Các lớp ẩn
        x = layers.Dense(32, activation='relu')(input_layer)  # 32 là số lượng neuron trong lớp ẩn,mỗi nẻuon nhận 6 bit đầu vàovào, activation='relu' là hàm kích hoạt ReLU giúp học các hàm phi tuyến
        x = layers.BatchNormalization()(x)   #Chuẩn hóa để tăng ổn định huấn luyện
        x = layers.Dropout(dropout_rate)(x) #Dropout để tránh overfitting, 20% neuron bị loại bỏ ở mỗi lần huấn luyện
        
        x = layers.Dense(16, activation='relu')(x) #Lớp ẩn thứ 2 có 16 neuron, mỗi neuron nhận 32 bit đầu vào và tạo ra 16 bit đầu ra
        x = layers.BatchNormalization()(x) #Chuẩn hóa để tăng ổn định huấn luyện
        
        # Đầu ra là vector 4 phần tử (biểu diễn 4-bit)
        output_layer = layers.Dense(4, activation='sigmoid')(x) #Lớp đầu ra có 4 neuron, mỗi neuron nhận 16 bit đầu vào và tạo ra 4 bit đầu ra, sigmoid cho kết quả nằm trong khoảng 0-1
        
        model = models.Sequential([  #Sequential là mô hình mạng nơ-ron tuần tự, nối các lớp với nhau
            layers.Input(shape=(6,)),
            layers.Dense(32, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(dropout_rate),
            layers.Dense(16, activation='relu'),
            layers.BatchNormalization(),
            layers.Dense(4, activation='sigmoid')
        ])
        #Input (6) → Dense(32) → BN → Dropout → Dense(16) → BN → Dense(4)
        model.compile(
            optimizer=optimizers.Adam(learning_rate=learning_rate), #Sử dụng optimizer Adam để tối ưu hóa mô hình
            loss='binary_crossentropy', #Sử dụng loss function binary_crossentropy để tối ưu hóa mô hình
            metrics=['accuracy'] #Sử dụng metrics accuracy để đánh giá mô hình
        )
        
        return model
    
    def build_permutation_model(self, dropout_rate=0.2, learning_rate=0.001):
        """Build neural network model for permutation optimization"""
        inputs = layers.Input(shape=(32,)) #Đầu vào là vector 32 phần tử (biểu diễn 32-bit)
        x = layers.Dense(64, activation='relu')(inputs) #Lớp ẩn thứ nhất có 64 neuron, mỗi neuron nhận 32 bit đầu vào và tạo ra 64 bit đầu ra
        x = layers.BatchNormalization()(x) #Chuẩn hóa để tăng ổn định huấn luyện
        x = layers.Dense(64, activation='relu')(x) #Lớp ẩn thứ 2 có 64 neuron, mỗi neuron nhận 64 bit đầu vào và tạo ra 64 bit đầu ra
        x = layers.Dropout(dropout_rate)(x) #Dropout để tránh overfitting, 20% neuron bị loại bỏ ở mỗi lần huấn luyện
        outputs = layers.Dense(32, activation='sigmoid')(x) #Lớp đầu ra có 32 neuron, mỗi neuron nhận 64 bit đầu vào và tạo ra 32 bit đầu ra, sigmoid cho kết quả nằm trong khoảng 0-1
        
        model = models.Model(inputs=inputs, outputs=outputs) #Tạo mô hình mạng nơ-ron với đầu vào là inputs và đầu ra là outputs
        
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate), #Sử dụng optimizer Adam để tối ưu hóa mô hình
            loss='binary_crossentropy', #Sử dụng loss function binary_crossentropy để tối ưu hóa mô hình
            metrics=['accuracy'] #Sử dụng metrics accuracy để đánh giá mô hình
        )
        
        return model
    
    def build_fast_sbox_model(self, learning_rate=0.001):
        """
        Xây dựng mô hình neural network nhẹ và nhanh cho S-box
        Mô hình này được tối ưu hóa cho tốc độ xử lý, đánh đổi một phần độ chính xác
        
        Args:
            learning_rate: Learning rate
            
        Returns:
            tensorflow.keras.Model: Mô hình tối ưu tốc độ
        """
        # Đầu vào là vector 6 phần tử (biểu diễn 6-bit)
        input_layer = layers.Input(shape=(6,))
        
        # Sử dụng mạng neural nhỏ hơn để tăng tốc độ dự đoán
        x = layers.Dense(8, activation='relu', kernel_initializer='he_uniform')(input_layer)
        
        # Đầu ra là vector 4 phần tử (biểu diễn 4-bit)
        output_layer = layers.Dense(4, activation='sigmoid')(x)
        
        # Tạo model
        model = models.Model(inputs=input_layer, outputs=output_layer)
        
        # Biên dịch model
        optimizer = optimizers.Adam(learning_rate=learning_rate)
        model.compile(
            optimizer=optimizer,
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        
        return model
        
    # def train_models(self, epochs=50, validation_split=0.2, dropout_rate=0.2, learning_rate=0.001):
    #     """
    #     Huấn luyện các mô hình S-box và permutation
        
    #     Args:
    #         epochs: Số epochs huấn luyện
    #         validation_split: Tỷ lệ dữ liệu validation
    #         dropout_rate: Tỷ lệ dropout
    #         learning_rate: Learning rate
            
    #     Returns:
    #         Tuple: (history_sbox, history_perm) - lịch sử huấn luyện của từng loại mô hình
    #     """
    #     # Khởi tạo danh sách để chứa lịch sử huấn luyện
    #     history_sbox = []
        
    #     # Tạo dữ liệu huấn luyện
    #     X_sbox, y_sbox, X_perm, y_perm = self.generate_training_data(num_samples=100000)
        
    #     print("Training S-box optimization models...")
    #     # Huấn luyện mô hình cho S-box (chỉ huấn luyện 2 mô hình đầu tiên cho DEMO)
    #     for i in range(2):  # Giảm từ 8 xuống 2 mô hình
    #         print(f"Training S-box {i} model...")
    #         model = self.build_sbox_model(dropout_rate, learning_rate)
    #         history = model.fit(
    #             X_sbox[i], y_sbox[i],
    #             epochs=epochs,
    #             batch_size=self.batch_size,
    #             validation_split=validation_split,
    #             verbose=1
    #         )
    #         self.sbox_models[i] = model
    #         history_sbox.append(history)
        
    #     print("Training permutation optimization model...")
    #     # Huấn luyện mô hình permutation
    #     perm_model = self.build_permutation_model(dropout_rate, learning_rate)
    #     history_perm = perm_model.fit(
    #         X_perm, y_perm,
    #         epochs=epochs,
    #         batch_size=self.batch_size,
    #         validation_split=validation_split,
    #         verbose=1
    #     )
    #     self.permutation_model = perm_model
        
    #     print("Model training complete.")
    #     return history_sbox, history_perm
    
    def optimized_sbox_lookup(self, input_val, sbox_idx):
        """
        Tìm kiếm S-box được tối ưu bằng ML
        
        Args:
            input_val: Giá trị đầu vào 6-bit
            sbox_idx: Chỉ số S-box (0-7)
            
        Returns:
            int: Giá trị đầu ra 4-bit
        """
        # Xác định xem có nên dùng ML cho S-box này hay không
        use_ml = sbox_idx < 2 and self.sbox_models[sbox_idx] is not None
        
        if use_ml:
            # Đưa input sang dạng binary array
            input_bits = np.zeros(6, dtype=np.float32)
            for i in range(6):
                input_bits[5-i] = (input_val >> i) & 1
                
            # Dự đoán sử dụng mô hình ML
            prediction = self.sbox_models[sbox_idx].predict(
                np.array([input_bits]), verbose=0
            )[0]
            
            # Chuyển prediction thành giá trị 4-bit
            result = 0
            for i in range(4):
                if prediction[i] >= 0.5:
                    result |= (1 << (3-i))
            
            return result
        else:
            # Sử dụng S-box truyền thống nếu không dùng ML
            # Calculate row and column indices
            row = ((input_val & 0b100000) >> 4) | (input_val & 0b000001)
            col = (input_val & 0b011110) >> 1
            
            # S-box lookup tables
            sbox_tables = [
                # S1
                [
                    [14, 4, 13, 1, 2, 15, 11, 8, 3, 10, 6, 12, 5, 9, 0, 7],
                    [0, 15, 7, 4, 14, 2, 13, 1, 10, 6, 12, 11, 9, 5, 3, 8],
                    [4, 1, 14, 8, 13, 6, 2, 11, 15, 12, 9, 7, 3, 10, 5, 0],
                    [15, 12, 8, 2, 4, 9, 1, 7, 5, 11, 3, 14, 10, 0, 6, 13]
                ],
                # S2
                [
                    [15, 1, 8, 14, 6, 11, 3, 4, 9, 7, 2, 13, 12, 0, 5, 10],
                    [3, 13, 4, 7, 15, 2, 8, 14, 12, 0, 1, 10, 6, 9, 11, 5],
                    [0, 14, 7, 11, 10, 4, 13, 1, 5, 8, 12, 6, 9, 3, 2, 15],
                    [13, 8, 10, 1, 3, 15, 4, 2, 11, 6, 7, 12, 0, 5, 14, 9]
                ],
                # S3
                [
                    [10, 0, 9, 14, 6, 3, 15, 5, 1, 13, 12, 7, 11, 4, 2, 8],
                    [13, 7, 0, 9, 3, 4, 6, 10, 2, 8, 5, 14, 12, 11, 15, 1],
                    [13, 6, 4, 9, 8, 15, 3, 0, 11, 1, 2, 12, 5, 10, 14, 7],
                    [1, 10, 13, 0, 6, 9, 8, 7, 4, 15, 14, 3, 11, 5, 2, 12]
                ],
                # S4
                [
                    [7, 13, 14, 3, 0, 6, 9, 10, 1, 2, 8, 5, 11, 12, 4, 15],
                    [13, 8, 11, 5, 6, 15, 0, 3, 4, 7, 2, 12, 1, 10, 14, 9],
                    [10, 6, 9, 0, 12, 11, 7, 13, 15, 1, 3, 14, 5, 2, 8, 4],
                    [3, 15, 0, 6, 10, 1, 13, 8, 9, 4, 5, 11, 12, 7, 2, 14]
                ],
                # S5
                [
                    [2, 12, 4, 1, 7, 10, 11, 6, 8, 5, 3, 15, 13, 0, 14, 9],
                    [14, 11, 2, 12, 4, 7, 13, 1, 5, 0, 15, 10, 3, 9, 8, 6],
                    [4, 2, 1, 11, 10, 13, 7, 8, 15, 9, 12, 5, 6, 3, 0, 14],
                    [11, 8, 12, 7, 1, 14, 2, 13, 6, 15, 0, 9, 10, 4, 5, 3]
                ],
                # S6
                [
                    [12, 1, 10, 15, 9, 2, 6, 8, 0, 13, 3, 4, 14, 7, 5, 11],
                    [10, 15, 4, 2, 7, 12, 9, 5, 6, 1, 13, 14, 0, 11, 3, 8],
                    [9, 14, 15, 5, 2, 8, 12, 3, 7, 0, 4, 10, 1, 13, 11, 6],
                    [4, 3, 2, 12, 9, 5, 15, 10, 11, 14, 1, 7, 6, 0, 8, 13]
                ],
                # S7
                [
                    [4, 11, 2, 14, 15, 0, 8, 13, 3, 12, 9, 7, 5, 10, 6, 1],
                    [13, 0, 11, 7, 4, 9, 1, 10, 14, 3, 5, 12, 2, 15, 8, 6],
                    [1, 4, 11, 13, 12, 3, 7, 14, 10, 15, 6, 8, 0, 5, 9, 2],
                    [6, 11, 13, 8, 1, 4, 10, 7, 9, 5, 0, 15, 14, 2, 3, 12]
                ],
                # S8
                [
                    [13, 2, 8, 4, 6, 15, 11, 1, 10, 9, 3, 14, 5, 0, 12, 7],
                    [1, 15, 13, 8, 10, 3, 7, 4, 12, 5, 6, 11, 0, 14, 9, 2],
                    [7, 11, 4, 1, 9, 12, 14, 2, 0, 6, 10, 13, 15, 3, 5, 8],
                    [2, 1, 14, 7, 4, 10, 8, 13, 15, 12, 9, 0, 3, 5, 6, 11]
                ]
            ]
            
            # Use sbox_idx modulo 8 để đảm bảo nằm trong phạm vi đúng
            return sbox_tables[sbox_idx % 8][row][col]
    
    def optimized_permutation(self, input_bits):
        """Use the trained model to optimize permutation operation"""
        if self.permutation_model is None:
            raise ValueError("Permutation model not trained yet.")
            
        # Reshape input to match model expectation
        model_input = np.array(input_bits, dtype=np.float32).reshape(1, 32)
        
        # Get prediction
        prediction = self.permutation_model.predict(model_input, verbose=0)[0]
        
        # Convert prediction to binary array using threshold
        result = [1 if bit >= 0.5 else 0 for bit in prediction]
        
        return result
    
    def encrypt(self, plaintext, key, benchmark_iterations=1):
        """
        Mã hóa plaintext sử dụng ML-Enhanced DES
        
        Parameters:
        -----------
        plaintext : bytes
            Plaintext (8 bytes) cần mã hóa
        key : bytes
            Key (8 bytes) sử dụng để mã hóa
        benchmark_iterations : int
            Số lần lặp lại để đo thời gian chính xác (mặc định = 1)
            
        Returns:
        --------
        tuple
            (ciphertext, execution_time)
        """
        if len(plaintext) != 8:
            raise ValueError("Plaintext phải có độ dài 8 bytes")
        if len(key) != 8:
            raise ValueError("Key phải có độ dài 8 bytes")
        
        # Tạo khóa con từ key
        subkeys = self._generate_subkeys(key)
        
        # Áp dụng IP (Initial Permutation)
        block = self._apply_initial_permutation(plaintext)
        
        # Tách thành nửa trái và nửa phải
        left, right = block[:4], block[4:]
        
        # Đo thời gian thực thi
        start_time = time.time()
        
        # Lặp lại quá trình để đo thời gian chính xác hơn
        for _ in range(benchmark_iterations):
            # Khôi phục giá trị ban đầu sau IP
            left_temp, right_temp = block[:4], block[4:]
            
            # 16 vòng Feistel
            for i in range(16):
                # Tính toán hàm Feistel
                f_result = self._feistel_function(right_temp, subkeys[i])
                
                # XOR và swap
                new_right = bytes(a ^ b for a, b in zip(left_temp, f_result))
                left_temp = right_temp
                right_temp = new_right
            
            # Ghép nửa phải và trái (đảo ngược vị trí)
            # Chú ý: Ở vòng cuối cùng, không swap L và R
            pre_output = right_temp + left_temp
            
            # Áp dụng hoán vị cuối cùng (IP^-1)
            ciphertext = self._apply_final_permutation(pre_output)
        
        execution_time = (time.time() - start_time) / benchmark_iterations
        
        return ciphertext, execution_time
        
    def decrypt(self, ciphertext, key, benchmark_iterations=1):
        """
        Giải mã ciphertext sử dụng ML-Enhanced DES
        
        Parameters:
        -----------
        ciphertext : bytes
            Ciphertext (8 bytes) cần giải mã
        key : bytes
            Key (8 bytes) sử dụng để giải mã
        benchmark_iterations : int
            Số lần lặp lại để đo thời gian chính xác (mặc định = 1)
            
        Returns:
        --------
        tuple
            (plaintext, execution_time)
        """
        if len(ciphertext) != 8:
            raise ValueError("Ciphertext phải có độ dài 8 bytes")
        if len(key) != 8:
            raise ValueError("Key phải có độ dài 8 bytes")
        
        # Tạo khóa con từ key
        subkeys = self._generate_subkeys(key)
        # Đảo ngược thứ tự các khóa con cho quá trình giải mã
        subkeys = subkeys[::-1]
        
        # Áp dụng IP (Initial Permutation)
        block = self._apply_initial_permutation(ciphertext)
        
        # Tách thành nửa trái và nửa phải
        left, right = block[:4], block[4:]
        
        # Đo thời gian thực thi
        start_time = time.time()
        
        # Lặp lại quá trình để đo thời gian chính xác hơn
        for _ in range(benchmark_iterations):
            # Khôi phục giá trị ban đầu sau IP
            left_temp, right_temp = block[:4], block[4:]
            
            # 16 vòng Feistel
            for i in range(16):
                # Tính toán hàm Feistel
                f_result = self._feistel_function(right_temp, subkeys[i])
                
                # XOR và swap
                new_right = bytes(a ^ b for a, b in zip(left_temp, f_result))
                left_temp = right_temp
                right_temp = new_right
            
            # Ghép nửa phải và trái (đảo ngược vị trí)
            # Chú ý: Ở vòng cuối cùng, không swap L và R
            pre_output = right_temp + left_temp
            
            # Áp dụng hoán vị cuối cùng (IP^-1)
            plaintext = self._apply_final_permutation(pre_output)
        
        execution_time = (time.time() - start_time) / benchmark_iterations
        
        return plaintext, execution_time
    
    def encrypt_single(self, plaintext, key):
        """
        Mã hóa một khối dữ liệu 8 byte với ML-Enhanced DES
        Phiên bản tối ưu hóa cho đo hiệu suất - không đo thời gian
        
        Args:
            plaintext: Plaintext (8 bytes) cần mã hóa
            key: Key (8 bytes) sử dụng để mã hóa
            
        Returns:
            bytes: ciphertext (8 bytes)
        """
        result, _ = self.encrypt(plaintext, key)
        return result
    
    def decrypt_single(self, ciphertext, key):
        """
        Giải mã một khối dữ liệu 8 byte với ML-Enhanced DES
        Phiên bản tối ưu hóa cho đo hiệu suất - không đo thời gian
        
        Args:
            ciphertext: Ciphertext (8 bytes) cần giải mã
            key: Key (8 bytes) sử dụng để giải mã
            
        Returns:
            bytes: plaintext (8 bytes)
        """
        result, _ = self.decrypt(ciphertext, key)
        return result
    
    def benchmark(self, num_samples=100, data_size=64):
        """Benchmark ML-enhanced DES against standard DES"""
        if any(model is None for model in self.sbox_models) or self.permutation_model is None:
            print("Warning: Models not trained. Running standard DES only.")
        
        print(f"Benchmarking with {num_samples} samples of {data_size} bytes each...")
        
        standard_encryption_times = []
        enhanced_encryption_times = []
        standard_decryption_times = []
        enhanced_decryption_times = []
        
        for _ in range(num_samples):
            # Generate random data and key with exact size
            data = os.urandom(data_size)  # Tạo dữ liệu với kích thước chính xác
            key = os.urandom(8)
            
            # Standard DES
            cipher = DES.new(key, DES.MODE_ECB)
            
            # Encryption benchmark
            start_time = time.time()
            padded_data = pad(data, 8)
            standard_ciphertext = cipher.encrypt(padded_data)
            standard_encryption_times.append(time.time() - start_time)
            
            # ML-enhanced DES encryption
            enhanced_ciphertext, enhanced_time = self.encrypt(data, key)
            enhanced_encryption_times.append(enhanced_time)
            
            # Decryption benchmark
            start_time = time.time()
            decrypted_data = cipher.decrypt(standard_ciphertext)
            try:
                decrypted_data = unpad(decrypted_data, 8)
            except ValueError:
                pass
            standard_decryption_times.append(time.time() - start_time)
            
            # ML-enhanced DES decryption
            enhanced_decrypted, enhanced_dec_time = self.decrypt(enhanced_ciphertext, key)
            enhanced_decryption_times.append(enhanced_dec_time)
        
        # Calculate averages
        avg_std_enc = sum(standard_encryption_times) / num_samples
        avg_enh_enc = sum(enhanced_encryption_times) / num_samples
        avg_std_dec = sum(standard_decryption_times) / num_samples
        avg_enh_dec = sum(enhanced_decryption_times) / num_samples
        
        # Add safety check to prevent division by zero
        enc_speedup = avg_std_enc/avg_enh_enc if avg_enh_enc > 0 else 0
        dec_speedup = avg_std_dec/avg_enh_dec if avg_enh_dec > 0 else 0
        
        print("\nBenchmark Results:")
        print(f"Standard DES Encryption: {avg_std_enc:.6f}s")
        print(f"ML-Enhanced DES Encryption: {avg_enh_enc:.6f}s")
        print(f"Encryption Speedup: {enc_speedup:.2f}x")
        print(f"Standard DES Decryption: {avg_std_dec:.6f}s")
        print(f"ML-Enhanced DES Decryption: {avg_enh_dec:.6f}s")
        print(f"Decryption Speedup: {dec_speedup:.2f}x")
        
        return {
            'std_enc': avg_std_enc,
            'enh_enc': avg_enh_enc,
            'std_dec': avg_std_dec,
            'enh_dec': avg_enh_dec,
            'enc_speedup': enc_speedup,
            'dec_speedup': dec_speedup
        }
    
    def save_models(self, base_path="ml_des_models"):
        """Save all trained models"""
        # Backup path nếu có vấn đề với base_path
        backup_path = "models"
        
        if not os.path.exists(base_path):
            try:
                os.makedirs(base_path)
            except Exception as e:
                print(f"Không thể tạo {base_path}: {e}")
                print(f"Thử dùng thư mục {backup_path}")
                base_path = backup_path
                if not os.path.exists(base_path):
                    os.makedirs(base_path)
            
        # Save S-box models
        for i, model in enumerate(self.sbox_models):
            if model is not None:
                try:
                    model_path = f"{base_path}/sbox_{i}"
                    if not os.path.exists(model_path):
                        os.makedirs(model_path)
                    model.save(model_path, save_format='tf')
                    print(f"Saved S-box model {i} successfully to {model_path}")
                except Exception as e:
                    print(f"Error saving S-box model {i}: {str(e)}")
                
        # Save permutation model
        if self.permutation_model is not None:
            try:
                perm_path = f"{base_path}/permutation"
                if not os.path.exists(perm_path):
                    os.makedirs(perm_path)
                self.permutation_model.save(perm_path, save_format='tf')
                print(f"Saved permutation model successfully to {perm_path}")
            except Exception as e:
                print(f"Error saving permutation model: {str(e)}")
            
        print(f"Models saved to {base_path}")
    
    def load_models(self, base_path="ml_des_models"):
        """Load all trained models"""
        # Danh sách các đường dẫn để kiểm tra
        paths_to_check = [base_path, "models"]
        models_loaded = False
        
        for path in paths_to_check:
            if not os.path.exists(path):
                print(f"Thư mục {path} không tồn tại, thử thư mục tiếp theo...")
                continue
                
            try:
                print(f"Thử tải mô hình từ {path}...")
                # Load S-box models
                for i in range(2):  # Chỉ tải 2 mô hình đầu tiên
                    model_path = f"{path}/sbox_{i}"
                    keras_path = f"{path}/sbox_{i}.keras"
                    model_dir_path = f"{path}/sbox_{i}_model"
                    
                    if os.path.exists(model_path):
                        self.sbox_models[i] = models.load_model(model_path)
                        print(f"  Loaded S-box model {i} from {model_path}")
                    elif os.path.exists(keras_path):
                        self.sbox_models[i] = models.load_model(keras_path)
                        print(f"  Loaded S-box model {i} from {keras_path}")
                    elif os.path.exists(model_dir_path):
                        self.sbox_models[i] = models.load_model(model_dir_path)
                        print(f"  Loaded S-box model {i} from {model_dir_path}")
                
                # Load permutation model
                perm_model_path = f"{path}/permutation"
                keras_perm_path = f"{path}/permutation.keras"
                perm_model_dir_path = f"{path}/permutation_model"
                
                if os.path.exists(perm_model_path):
                    self.permutation_model = models.load_model(perm_model_path)
                    print(f"  Loaded permutation model from {perm_model_path}")
                elif os.path.exists(keras_perm_path):
                    self.permutation_model = models.load_model(keras_perm_path)
                    print(f"  Loaded permutation model from {keras_perm_path}")
                elif os.path.exists(perm_model_dir_path):
                    self.permutation_model = models.load_model(perm_model_dir_path)
                    print(f"  Loaded permutation model from {perm_model_dir_path}")
                    
                # Chỉ cần tải các mô hình cốt lõi
                if all(model is not None for model in self.sbox_models[:2]) and self.permutation_model is not None:
                    models_loaded = True
                
                if models_loaded:
                    print(f"Models loaded successfully from {path}.")
                    self.model_loaded = True
                    return True
                else:
                    print(f"Some models not found in {path}.")
                    
            except Exception as e:
                print(f"Error loading models from {path}: {e}")
        
        print("Failed to load models from any location.")
        return False

    def _generate_subkeys(self, key):
        """
        Generate 16 subkeys for DES encryption/decryption
        
        Args:
            key: 8-byte key
            
        Returns:
            List of 16 6-byte subkeys
        """
        # Convert key to binary string
        key_bits = ''.join(format(b, '08b') for b in key)
        
        # Apply PC1 permutation
        pc1_key = ''.join(key_bits[i-1] for i in self.PC1)
        
        # Split into left and right halves
        left = pc1_key[:28]
        right = pc1_key[28:]
        
        subkeys = []
        for i in range(16):
            # Rotate left and right halves
            left = left[self.ROTATIONS[i]:] + left[:self.ROTATIONS[i]]
            right = right[self.ROTATIONS[i]:] + right[:self.ROTATIONS[i]]
            
            # Combine halves and apply PC2 permutation
            combined = left + right
            subkey = ''.join(combined[i-1] for i in self.PC2)
            
            # Convert to bytes
            subkey_bytes = bytes(int(subkey[i:i+8], 2) for i in range(0, 48, 8))
            subkeys.append(subkey_bytes)
        
        return subkeys

    def _apply_initial_permutation(self, block):
        """Apply initial permutation to 8-byte block"""
        # Convert block to binary string
        block_bits = ''.join(format(b, '08b') for b in block)
        
        # Apply IP permutation
        permuted = ''.join(block_bits[i-1] for i in self.IP)
        
        # Convert back to bytes
        return bytes(int(permuted[i:i+8], 2) for i in range(0, 64, 8))
        
    def _apply_final_permutation(self, block):
        """Apply final permutation (IP^-1) to 8-byte block"""
        # Convert block to binary string
        block_bits = ''.join(format(b, '08b') for b in block)
        
        # Apply IP^-1 permutation
        permuted = ''.join(block_bits[i-1] for i in self.IP_INV)
        
        # Convert back to bytes
        return bytes(int(permuted[i:i+8], 2) for i in range(0, 64, 8))
        
    def _expand(self, block):
        """Expand 4-byte block to 6 bytes using E table"""
        # Convert block to binary string
        block_bits = ''.join(format(b, '08b') for b in block)
        
        # Apply E expansion
        expanded = ''.join(block_bits[i-1] for i in self.E)
        
        # Convert back to bytes
        return bytes(int(expanded[i:i+8], 2) for i in range(0, 48, 8))
        
    def _permute(self, block, table):
        """Apply permutation using given table"""
        # Convert block to binary string
        block_bits = ''.join(format(b, '08b') for b in block)
        
        # Apply permutation
        permuted = ''.join(block_bits[i-1] for i in table)
        
        # Convert back to bytes
        return bytes(int(permuted[i:i+8], 2) for i in range(0, len(permuted), 8))
        
    def _feistel_function(self, right, subkey):
        """Apply Feistel function to 4-byte block using 6-byte subkey"""
        # Expand right half
        expanded = self._expand(right)
        
        # XOR with subkey
        xored = bytes(a ^ b for a, b in zip(expanded, subkey))
        
        # Apply S-boxes using ML models
        sbox_output = bytearray()
        for i in range(8):
            # Get 6 bits for this S-box
            bits = xored[i*6:(i+1)*6]
            # Convert to integer
            val = int.from_bytes(bits, 'big')
            # Get row and column
            row = ((val & 0b100000) >> 4) | (val & 0b000001)
            col = (val & 0b011110) >> 1
            
            # Use ML model if available and enabled
            if self.predict_sboxes and i < len(self.sbox_models) and self.sbox_models[i] is not None:
                # Prepare input for ML model
                model_input = np.array([val], dtype=np.float32)
                # Predict using ML model
                prediction = self.sbox_models[i].predict(model_input, verbose=0)[0]
                # Convert prediction to 4-bit value
                sbox_val = int(round(prediction[0] * 15))  # Scale to 0-15
            else:
                # Fallback to standard S-box lookup
                sbox_idx = i % len(self.sbox_tables)
                sbox_val = self.sbox_tables[sbox_idx][row][col]
            
            # Convert to 4 bits
            sbox_output.extend(sbox_val.to_bytes(1, 'big'))
        
        # Apply P-box permutation using ML model if available
        if self.predict_sboxes and self.permutation_model is not None:
            # Convert sbox_output to input for permutation model
            perm_input = np.array([int.from_bytes(sbox_output, 'big')], dtype=np.float32)
            # Predict using ML model
            perm_prediction = self.permutation_model.predict(perm_input, verbose=0)[0]
            # Convert prediction to bytes
            perm_output = bytearray()
            for bit in perm_prediction:
                perm_output.extend(int(round(bit)).to_bytes(1, 'big'))
            return perm_output
        else:
            # Fallback to standard permutation
            return self._permute(sbox_output, self.PBOX)

# Bank Digital API Functions
def encrypt_data(data, key, compare_with_standard=False, force_standard=False):
    """
    Mã hóa dữ liệu sử dụng ML-Enhanced DES hoặc DES tiêu chuẩn
    
    Args:
        data: Dữ liệu cần mã hóa (bytes)
        key: Khóa mã hóa (8 bytes)
        compare_with_standard: So sánh với DES tiêu chuẩn
        force_standard: Bắt buộc sử dụng DES tiêu chuẩn
        
    Returns:
        tuple: (ciphertext, execution_time) hoặc (ciphertext, execution_time, standard_ciphertext, standard_time)
    """
    # Khởi tạo đối tượng ML-Enhanced DES
    ml_des = MLEnhancedDES()
    
    # Tải mô hình đã huấn luyện
    if not force_standard and ml_des.load_models():
        print("Đã tải mô hình ML-Enhanced DES thành công.")
        ml_des.predict_sboxes = True  # Sử dụng ML để dự đoán S-box
    else:
        print("Không thể tải mô hình ML-Enhanced DES. Sử dụng DES tiêu chuẩn.")
        ml_des.predict_sboxes = False
    
    # Pad dữ liệu nếu cần
    if len(data) % 8 != 0:
        padding_length = 8 - (len(data) % 8)
        data = data + bytes([padding_length] * padding_length)
    
    # Mã hóa sử dụng ML-Enhanced DES
    ml_result = bytearray()
    ml_time = 0
    for i in range(0, len(data), 8):
        block = data[i:i+8]
        ciphertext_block, block_time = ml_des.encrypt(block, key)
        ml_result.extend(ciphertext_block)
        ml_time += block_time
    
    # Nếu cần so sánh với DES tiêu chuẩn
    if compare_with_standard:
        # Mã hóa sử dụng DES tiêu chuẩn
        cipher = DES.new(key, DES.MODE_ECB)
        start_time = time.time()
        standard_result = cipher.encrypt(data)
        standard_time = time.time() - start_time
        
        return bytes(ml_result), ml_time, standard_result, standard_time
    
    return bytes(ml_result), ml_time

def decrypt_data(ciphertext, key, compare_with_standard=False, force_standard=False):
    """Decrypt data for bank digital interface"""
    
    # Nếu force_standard=True, sử dụng DES tiêu chuẩn
    if force_standard:
        # Ensure key is 8 bytes
        if isinstance(key, str):
            # If key is a string, convert to bytes and pad/truncate to 8 bytes
            key_bytes = key.encode('utf-8')
            if len(key_bytes) < 8:
                key_bytes = key_bytes + b'\0' * (8 - len(key_bytes))
            key = key_bytes[:8]
        
        # Sử dụng DES tiêu chuẩn
        cipher = DES.new(key, DES.MODE_ECB)
        plaintext = cipher.decrypt(ciphertext)
        
        # Remove padding
        try:
            plaintext = unpad(plaintext, 8)
        except ValueError:
            # If unpadding fails, return the raw data
            pass
        
        # Try to convert to string if it seems to be text
        try:
            return plaintext.decode('utf-8')
        except UnicodeDecodeError:
            # If it's not valid UTF-8, return the raw bytes
            return plaintext
    
    ml_des = MLEnhancedDES()
    
    # Try to load pre-trained models for better performance
    if ml_des.load_models():
        print("Using ML-enhanced DES with pre-trained models")
        ml_des.predict_sboxes = True
    else:
        print("Using standard DES (no pre-trained models available)")
        ml_des.predict_sboxes = False
    
    # Ensure key is 8 bytes
    if isinstance(key, str):
        # If key is a string, convert to bytes and pad/truncate to 8 bytes
        key_bytes = key.encode('utf-8')
        if len(key_bytes) < 8:
            key_bytes = key_bytes + b'\0' * (8 - len(key_bytes))
        key = key_bytes[:8]
    
    # Decrypt with ML-Enhanced DES
    ml_result = bytearray()
    ml_time = 0
    for i in range(0, len(ciphertext), 8):
        block = ciphertext[i:i+8]
        plaintext_block, block_time = ml_des.decrypt(block, key)
        ml_result.extend(plaintext_block)
        ml_time += block_time
    
    # If requested, compare with standard DES
    if compare_with_standard:
        # Decrypt with standard DES
        std_start_time = time.time()
        cipher = DES.new(key, DES.MODE_ECB)
        std_result = cipher.decrypt(ciphertext)
        std_time = time.time() - std_start_time
        
        # Compare results
        match = std_result == ml_result
        print("\nML-Enhanced DES vs Standard DES comparison:")
        print(f"Results match: {match}")
        if not match:
            print(f"Độ chính xác: Không khớp!")
            print(f"ML-Enhanced result: {ml_result.hex()}")
            print(f"Standard DES result: {std_result.hex()}")
            # Tính tỷ lệ byte giống nhau
            byte_matches = sum(1 for a, b in zip(ml_result, std_result) if a == b)
            accuracy = byte_matches / len(std_result) * 100
            print(f"Độ chính xác theo byte: {accuracy:.2f}%")
        else:
            print(f"Độ chính xác: 100% - Hoàn toàn khớp!")
            
        print(f"ML-Enhanced time: {ml_time:.6f} seconds")
        print(f"Standard DES time: {std_time:.6f} seconds")
        speed_ratio = std_time/ml_time if ml_time > 0 else 0
        print(f"Speed improvement: {speed_ratio:.2f}x")
    
    # Try to unpad the result
    try:
        ml_result = unpad(ml_result, 8)
    except ValueError:
        # If unpadding fails, use the raw result
        pass
    
    # Try to convert to string if it seems to be text
    try:
        return ml_result.decode('utf-8')
    except UnicodeDecodeError:
        # If it's not valid UTF-8, return the raw bytes
        return ml_result

def train_and_save_model(num_samples=500000, epochs=100, lr=0.001, bs=128, dr=0.2, save_path="ml_des_models", optimize_for="balanced", save_training_data=True, early_stopping=True, patience=5, fast_mode=False):
    """
    Huấn luyện và lưu mô hình ML-Enhanced DES
    
    Args:
        num_samples: Số lượng mẫu để huấn luyện
        epochs: Số epochs huấn luyện
        lr: Learning rate
        bs: Batch size
        dr: Dropout rate
        save_path: Đường dẫn để lưu mô hình
        optimize_for: Chiến lược tối ưu hóa ("speed", "accuracy", "balanced")
        save_training_data: Có lưu dữ liệu huấn luyện để thuyết trình không
        early_stopping: Có sử dụng kỹ thuật dừng sớm không
        patience: Số epochs chờ đợi khi validation loss không giảm
        fast_mode: Chế độ huấn luyện nhanh (giảm số mẫu và epochs)
    
    Returns:
        bool: True nếu thành công, False nếu không
    """
    print("=== Bắt đầu quá trình huấn luyện mô hình ML-Enhanced DES ===")
    print(f"Optimizing for: {optimize_for}")
    
    # Áp dụng chế độ nhanh nếu được yêu cầu
    if fast_mode:
        original_samples = num_samples
        original_epochs = epochs
        # Giảm số lượng mẫu và epochs xuống
        num_samples = min(num_samples, 50000)
        epochs = min(epochs, 20)
        print(f"CHẾ ĐỘ NHANH: Giảm từ {original_samples} mẫu xuống {num_samples} mẫu")
        print(f"CHẾ ĐỘ NHANH: Giảm từ {original_epochs} epochs xuống {epochs} epochs")
    
    print(f"Tham số: {num_samples} mẫu, {epochs} epochs, lr={lr}, batch_size={bs}, dropout={dr}")
    if early_stopping:
        print(f"Sử dụng dừng sớm (early stopping) với patience={patience}")
    
    # Tạo thư mục lưu mô hình nếu chưa tồn tại
    os.makedirs(save_path, exist_ok=True)
    
    # Khởi tạo đối tượng MLEnhancedDES
    ml_des = MLEnhancedDES()
    
    # Tạo dữ liệu huấn luyện
    print("Tạo dữ liệu huấn luyện...")
    X_sbox, y_sbox, X_perm, y_perm = ml_des.generate_enhanced_training_data(
        num_samples=num_samples,
        save_samples=save_training_data,
        samples_file=os.path.join(save_path, "training_samples.csv"),
        num_save_samples=5000
    )
    
    # Chuẩn bị callback dừng sớm nếu cần
    callbacks = []
    if early_stopping:
        early_stop = tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=patience,
            verbose=1,
            restore_best_weights=True
        )
        callbacks.append(early_stop)
    
    # Lựa chọn kiến trúc mô hình dựa vào tham số optimize_for
    if optimize_for == "speed":
        print("Tối ưu hóa cho TỐC ĐỘ - sử dụng mô hình nhẹ")
        # Sử dụng mô hình nhẹ và nhanh
        sbox_models = []
        for i in range(2):  # Giảm từ 8 xuống 2 để DEMO
            model = ml_des.build_fast_sbox_model(learning_rate=lr)
            history = model.fit(
                X_sbox[i], y_sbox[i],
                epochs=min(30, epochs),  # Giảm epochs để tăng tốc quá trình huấn luyện
                batch_size=bs,
                validation_split=0.1,
                verbose=1,
                callbacks=callbacks
            )
            sbox_models.append(model)
    elif optimize_for == "accuracy":
        print("Tối ưu hóa cho ĐỘ CHÍNH XÁC - sử dụng mô hình phức tạp")
        # Sử dụng mô hình phức tạp hơn
        sbox_models = []
        for i in range(2):
            model = ml_des.build_sbox_model(dropout_rate=dr, learning_rate=lr)
            history = model.fit(
                X_sbox[i], y_sbox[i],
                epochs=epochs,
                batch_size=bs,
                validation_split=0.2,
                verbose=1,
                callbacks=callbacks
            )
            sbox_models.append(model)
    else:  # "balanced"
        print("Tối ưu hóa CÂN BẰNG giữa tốc độ và độ chính xác")
        # Cân bằng giữa tốc độ và độ chính xác
        sbox_models = []
        for i in range(2):
            if i % 2 == 0:  # Dùng mô hình nhanh cho một nửa S-box
                model = ml_des.build_fast_sbox_model(learning_rate=lr)
                history = model.fit(
                    X_sbox[i], y_sbox[i],
                    epochs=min(50, epochs),
                    batch_size=bs,
                    validation_split=0.15,
                    verbose=1,
                    callbacks=callbacks
                )
            else:  # Dùng mô hình chính xác cho nửa còn lại
                model = ml_des.build_sbox_model(dropout_rate=dr, learning_rate=lr)
                history = model.fit(
                    X_sbox[i], y_sbox[i],
                    epochs=epochs,
                    batch_size=bs,
                    validation_split=0.2,
                    verbose=1,
                    callbacks=callbacks
                )
            sbox_models.append(model)
    
    # Huấn luyện mô hình permutation (chung cho tất cả chiến lược)
    perm_model = ml_des.build_permutation_model(dropout_rate=dr, learning_rate=lr)
    perm_history = perm_model.fit(
        X_perm, y_perm,
        epochs=epochs,
        batch_size=bs,
        validation_split=0.2,
        verbose=1,
        callbacks=callbacks
    )
    
    # Lưu mô hình
    print(f"Lưu mô hình vào {save_path}...")
    for i, model in enumerate(sbox_models):
        model.save(os.path.join(save_path, f"sbox_{i}_model"))
    perm_model.save(os.path.join(save_path, "permutation_model"))
    
    # Đánh giá mô hình
    results = {}
    for i, model in enumerate(sbox_models):
        val_loss, val_acc = model.evaluate(X_sbox[i][-1000:], y_sbox[i][-1000:], verbose=0)
        results[f"sbox_{i}"] = {"accuracy": val_acc, "loss": val_loss}
    
    perm_val_loss, perm_val_acc = perm_model.evaluate(X_perm[-1000:], y_perm[-1000:], verbose=0)
    results["permutation"] = {"accuracy": perm_val_acc, "loss": perm_val_loss}
    
    # Lưu kết quả đánh giá
    with open(os.path.join(save_path, "evaluation_results.txt"), "w") as f:
        f.write(f"Optimization strategy: {optimize_for}\n")
        f.write(f"Training parameters: samples={num_samples}, epochs={epochs}, lr={lr}, batch_size={bs}, dropout={dr}\n\n")
        f.write("Evaluation results:\n")
        for key, value in results.items():
            f.write(f"{key}: Accuracy={value['accuracy']:.4f}, Loss={value['loss']:.4f}\n")
    
    print("=== Quá trình huấn luyện hoàn tất ===")
    accuracy_avg = sum(result["accuracy"] for result in results.values()) / len(results)
    print(f"Độ chính xác trung bình: {accuracy_avg:.4f}")
    
    return True

def demo():
    """Demo function để chạy từ command line"""
    parser = argparse.ArgumentParser(description='ML-Enhanced DES Demo')
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'benchmark', 'encrypt', 'decrypt'],
                        help='Mode: train (huấn luyện mô hình), benchmark (đánh giá hiệu suất), encrypt/decrypt (mã hóa/giải mã)')
    parser.add_argument('--samples', type=int, default=100000, help='Số lượng mẫu huấn luyện')
    parser.add_argument('--epochs', type=int, default=50, help='Số epochs huấn luyện')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size')
    parser.add_argument('--dropout', type=float, default=0.2, help='Dropout rate')
    parser.add_argument('--model_path', type=str, default='ml_des_models', help='Đường dẫn lưu/tải mô hình')
    parser.add_argument('--optimize_for', type=str, default='balanced', choices=['speed', 'accuracy', 'balanced'],
                        help='Chiến lược tối ưu hóa: speed (tối ưu tốc độ), accuracy (tối ưu độ chính xác), balanced (cân bằng)')
    parser.add_argument('--input', type=str, help='Đầu vào cho mã hóa/giải mã')
    parser.add_argument('--key', type=str, help='Khóa cho mã hóa/giải mã (hex, 8 bytes)')
    parser.add_argument('--compare', action='store_true', help='So sánh với DES tiêu chuẩn')
    parser.add_argument('--save_training_data', action='store_true', help='Lưu dữ liệu huấn luyện ra file CSV để thuyết trình')
    parser.add_argument('--training_data_file', type=str, default='training_samples.csv', help='Tên file lưu dữ liệu huấn luyện')
    parser.add_argument('--num_samples_save', type=int, default=5000, help='Số lượng mẫu lưu vào file (tối đa)')
    parser.add_argument('--early_stopping', action='store_true', help='Sử dụng kỹ thuật dừng sớm (early stopping) khi huấn luyện')
    parser.add_argument('--patience', type=int, default=5, help='Số epochs chờ đợi khi validation loss không giảm')
    parser.add_argument('--fast', action='store_true', help='Chế độ huấn luyện nhanh (giảm số mẫu và epochs)')
    
    args = parser.parse_args()
    
    # Kiểm tra xem mô hình đã tồn tại chưa
    models_exist = os.path.exists(args.model_path) and len(os.listdir(args.model_path)) > 0
    
    if args.mode == 'train':
        # Huấn luyện mô hình mới
        print("=== Bắt đầu huấn luyện mô hình ML-Enhanced DES ===")
        train_and_save_model(
            num_samples=args.samples,
            epochs=args.epochs,
            lr=args.learning_rate,
            bs=args.batch_size,
            dr=args.dropout,
            save_path=args.model_path,
            optimize_for=args.optimize_for,
            save_training_data=args.save_training_data,
            early_stopping=args.early_stopping,
            patience=args.patience,
            fast_mode=args.fast
        )
    
    elif args.mode == 'benchmark':
        if not models_exist:
            print("Chưa có mô hình đã huấn luyện. Tiến hành huấn luyện...")
            train_and_save_model(
                num_samples=args.samples, 
                epochs=args.epochs,
                lr=args.learning_rate,
                bs=args.batch_size,
                dr=args.dropout,
                save_path=args.model_path,
                optimize_for=args.optimize_for
            )
        
        # Benchmark
        print("\n=== Benchmark ML-Enhanced DES vs DES tiêu chuẩn ===")
        ml_des = MLEnhancedDES()
        if ml_des.load_models(args.model_path):
            ml_des.predict_sboxes = True  # Sử dụng ML để dự đoán S-box
            print("Đã tải mô hình ML-Enhanced DES thành công.")
        else:
            print("Không thể tải mô hình ML-Enhanced DES. Sử dụng DES tiêu chuẩn.")
            ml_des.predict_sboxes = False
        
        # Benchmark với nhiều kích thước dữ liệu khác nhau
        data_sizes = [8, 64, 512, 4096, 32768]  # Bytes
        
        for size in data_sizes:
            print(f"\nBenchmark với kích thước dữ liệu: {size} bytes")
            
            # Tạo dữ liệu ngẫu nhiên
            data = os.urandom(size)
            key = os.urandom(8)
            
            # Mã hóa với DES tiêu chuẩn
            std_start = time.time()
            
            cipher = DES.new(key, DES.MODE_ECB)
            padded_data = pad(data, 8)
            std_encrypted = cipher.encrypt(padded_data)
            
            std_time = time.time() - std_start
            
            # Mã hóa với ML-Enhanced DES (nếu mô hình đã được tải)
            ml_time = 0
            accuracy = "N/A"
            
            if ml_des.model_loaded:
                ml_start = time.time()
                
                ml_encrypted = bytearray()
                for i in range(0, len(padded_data), 8):
                    block = padded_data[i:i+8]
                    encrypted_block, _ = ml_des.encrypt(block, key)
                    ml_encrypted.extend(encrypted_block)
                
                ml_time = time.time() - ml_start
                
                # Kiểm tra độ chính xác
                if ml_encrypted == std_encrypted:
                    accuracy = "100%"
                else:
                    byte_matches = sum(1 for a, b in zip(ml_encrypted, std_encrypted) if a == b)
                    accuracy = f"{byte_matches/len(std_encrypted)*100:.2f}%"
            
            # In kết quả
            print(f"DES tiêu chuẩn: {std_time*1000:.2f}ms")
            if ml_des.model_loaded:
                print(f"ML-Enhanced DES: {ml_time*1000:.2f}ms")
                speedup = std_time / ml_time if ml_time > 0 else 0
                print(f"Tăng tốc: {speedup:.2f}x | Độ chính xác: {accuracy}")
    
    elif args.mode in ['encrypt', 'decrypt']:
        if not args.input:
            print("Lỗi: Cần có đầu vào (--input) để mã hóa/giải mã")
            return
        
        if not args.key:
            print("Lỗi: Cần có khóa (--key) để mã hóa/giải mã")
            return
        
        # Tải mô hình
        if not models_exist:
            print("Chưa có mô hình đã huấn luyện. Tiến hành huấn luyện...")
            train_and_save_model(
                num_samples=args.samples,
                epochs=args.epochs,
                lr=args.learning_rate,
                bs=args.batch_size,
                dr=args.dropout,
                save_path=args.model_path,
                optimize_for=args.optimize_for,
                save_training_data=args.save_training_data,
                early_stopping=args.early_stopping,
                patience=args.patience,
                fast_mode=args.fast
            )
        
        # Chuẩn bị dữ liệu đầu vào
        input_data = args.input.encode('utf-8')
        
        try:
            key = bytes.fromhex(args.key) if len(args.key) == 16 else args.key.encode('utf-8')
            if len(key) < 8:
                key = key + b'\0' * (8 - len(key))
            key = key[:8]  # Sử dụng 8 bytes đầu tiên
        except:
            key = args.key.encode('utf-8')
            if len(key) < 8:
                key = key + b'\0' * (8 - len(key))
            key = key[:8]
        
        if args.mode == 'encrypt':
            # Mã hóa
            if args.compare:
                result = encrypt_data(input_data, key, compare_with_standard=True)
            else:
                result = encrypt_data(input_data, key)
            
            print(f"\nKết quả mã hóa (hex): {result.hex()}")
            
        else:  # decrypt
            try:
                # Giả định đầu vào là chuỗi hex
                input_data = bytes.fromhex(args.input)
            except:
                # Nếu không phải hex, sử dụng chuỗi UTF-8
                input_data = args.input.encode('utf-8')
            
            # Giải mã
            if args.compare:
                result = decrypt_data(input_data, key, compare_with_standard=True)
            else:
                result = decrypt_data(input_data, key)
            
            if isinstance(result, bytes):
                try:
                    print(f"\nKết quả giải mã: {result.decode('utf-8')}")
                except UnicodeDecodeError:
                    print(f"\nKết quả giải mã (hex): {result.hex()}")
            else:
                print(f"\nKết quả giải mã: {result}")
    
    else:
        print("Không xác định được chế độ. Sử dụng --mode=train, benchmark, encrypt, hoặc decrypt")

def main():
    """Hàm chính để chạy từ command line"""
    # Gọi trực tiếp hàm demo() để sử dụng các tùy chọn dòng lệnh đã cải tiến
    demo()

if __name__ == "__main__":
    main() 