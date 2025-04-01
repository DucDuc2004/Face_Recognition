import os
import cv2
import json
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split

# Load Haar Cascade cho nhận diện khuôn mặt
haar_cascade = cv2.CascadeClassifier("Face_Recognition/model/haarcascade_frontalface_default.xml")

# Hàm trích xuất đặc trưng Haar
def extract_haar_features(img):
    features = haar_cascade.detectMultiScale(img, scaleFactor=1.1, minNeighbors=5)
    
    # Tạo một feature map trống
    feature_map = np.zeros_like(img, dtype=np.float32)

    for (x, y, w, h) in features:
        feature_map[y:y+h, x:x+w] = 255  # Đánh dấu vùng phát hiện khuôn mặt

    return feature_map / 255.0  # Chuẩn hóa về [0,1]

# Đọc danh sách nhãn từ file JSON
with open("Face_Recognition/src/label_map.json", "r") as f:
    label_dict = json.load(f)

# Đường dẫn đến dataset
dataset_path = "Face_Recognition/data"

# Load dữ liệu với đặc trưng Haar
X, y = [], []

for folder, label_id in label_dict.items():
    folder_path = os.path.join(dataset_path, folder)
    if os.path.isdir(folder_path):
        for img_name in os.listdir(folder_path):
            img_path = os.path.join(folder_path, img_name)
            img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)  # Đọc ảnh gốc (ảnh xám)

            if img is not None:
                img = cv2.resize(img, (200, 200))  # Resize ảnh
                haar_features = extract_haar_features(img)  # Trích xuất đặc trưng Haar
                
                X.append(haar_features)
                y.append(label_id)

# Chuyển đổi dữ liệu
X = np.array(X, dtype=np.float32)[..., np.newaxis]  # (số lượng ảnh, 200, 200, 1)
y = np.array(y, dtype=np.int32)

# Chia tập train/test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

# Data augmentation
datagen = ImageDataGenerator(
    rotation_range=10,  
    width_shift_range=0.1,  
    height_shift_range=0.1, 
    horizontal_flip=True    
)
datagen.fit(X_train)

# Xây dựng mô hình CNN
model = keras.Sequential([
    layers.Conv2D(32, (3,3), activation='relu', input_shape=(200, 200, 1)),
    layers.BatchNormalization(),
    layers.MaxPooling2D((2,2)),
    
    layers.Conv2D(64, (3,3), activation='relu'),
    layers.BatchNormalization(),
    layers.MaxPooling2D((2,2)),
    
    layers.Conv2D(128, (3,3), activation='relu'),
    layers.BatchNormalization(),
    layers.MaxPooling2D((2,2)),
    
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(len(label_dict), activation='softmax')
])

# Compile model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train model với dữ liệu được tăng cường
history = model.fit(datagen.flow(X_train, y_train, batch_size=32),
                    validation_data=(X_test, y_test),
                    epochs=50)

# Lưu model
model.save("model_face_recognition_haar.keras")

# Vẽ biểu đồ Training Loss & Accuracy
plt.figure(figsize=(12, 5))

# Accuracy Plot
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train Acc')
plt.plot(history.history['val_accuracy'], label='Val Acc')
plt.title("Training & Validation Accuracy")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.legend()

# Loss Plot
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.title("Training & Validation Loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()

# Lưu biểu đồ vào file
plt.savefig("result_training_haar.png")
plt.show()
