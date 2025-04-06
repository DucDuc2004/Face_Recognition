import cv2
import os
import json
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # Tắt GPU, dùng CPU
# Load model đã train (chú ý tên file model)
model = tf.keras.models.load_model("Face_Recognition/model/model.keras")

# Load danh sách nhãn từ file JSON
with open("Face_Recognition/src/label_map.json", "r") as f:
    label_dict = json.load(f)

# Đảo ngược danh sách nhãn: {0: "20224403_Le_Duc_Anh", 1: "20224409_Tran_Cong_Hiep", ...}
reverse_label_dict = {v: k for k, v in label_dict.items()}

# Load Haar Cascade để phát hiện khuôn mặt
face_cascade = cv2.CascadeClassifier("Face_Recognition/model/haarcascade_frontalface_default.xml")

# Hàm nhận diện khuôn mặt
def recognize_face(image_path):
    img = cv2.imread(image_path)
    
    if img is None:
        print("Lỗi: Không thể đọc ảnh.")
        return
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Phát hiện khuôn mặt
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(50, 50))

    if len(faces) == 0:
        print("Không phát hiện khuôn mặt!")
        return

    for (x, y, w, h) in faces:
        face_roi = gray[y:y+h, x:x+w]  # Cắt vùng mặt

        # Resize và chuẩn hóa ảnh
        face_resized = cv2.resize(face_roi, (200, 200)).astype(np.float32) / 255.0
        face_resized = np.expand_dims(face_resized, axis=(0, -1))  # (1, 200, 200, 1)

        # Tạo đặc trưng Haar bằng cách dùng lại ảnh xám resize (tạm thời)
        haar_feature = face_resized.copy()  # Hoặc có thể sinh từ feature extractor riêng nếu có

        # Dự đoán với 2 input
        predictions = model.predict({
            "input_image": face_resized,
            "haar_feature": haar_feature
        })

        predicted_label = np.argmax(predictions)
        confidence = np.max(predictions) * 100

        recognized_person = reverse_label_dict.get(predicted_label, "Không xác định")

        label_text = f"{recognized_person} ({confidence:.2f}%)"
        cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
        
        # Tăng kích thước font và độ dày đường viền
        font_scale = 1.0  # Font size
        font_thickness = 2  # Thickness of text
        cv2.putText(img, label_text, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 255, 0), font_thickness)

        print(f"Nhận diện: {recognized_person} (Độ chính xác: {confidence:.2f}%)")

    # Resize ảnh nếu cần
    img_display = cv2.resize(img, (1538, 2048))

    # Chuyển từ BGR (OpenCV) sang RGB (matplotlib)
    img_rgb = cv2.cvtColor(img_display, cv2.COLOR_BGR2RGB)

    # Hiển thị bằng matplotlib
    plt.imshow(img_rgb)
    plt.title("Face Recognition")
    plt.axis("off")
    plt.show()

# Đường dẫn ảnh cần nhận diện
image_path = "Face_Recognition/test/hiep1.jpg"

# Chạy nhận diện
recognize_face(image_path)
