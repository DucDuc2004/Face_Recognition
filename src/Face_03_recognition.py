import cv2
import json
import numpy as np
import tensorflow as tf

# Load model đã train
model = tf.keras.models.load_model("Face_Recognition/model/model_face_recognition1.keras")

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
        face_resized = cv2.resize(face_roi, (200, 200))
        face_resized = face_resized.astype(np.float32) / 255.0
        face_resized = np.expand_dims(face_resized, axis=-1)  # Thêm kênh màu (200, 200, 1)
        face_resized = np.expand_dims(face_resized, axis=0)   # Thêm batch dimension (1, 200, 200, 1)

        # Dự đoán
        predictions = model.predict(face_resized)
        predicted_label = np.argmax(predictions)
        confidence = np.max(predictions) * 100  # Độ chính xác %

        # Tra cứu ID_Ho_Ten
        recognized_person = reverse_label_dict.get(predicted_label, "Không xác định")

        # Hiển thị thông tin lên ảnh
        label_text = f"{recognized_person} ({confidence:.2f}%)"
        cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(img, label_text, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 2)

        print(f"Nhận diện: {recognized_person} (Độ chính xác: {confidence:.2f}%)")

    # Resize ảnh chỉ để hiển thị (giữ nguyên vị trí box)
    img_display = cv2.resize(img, (960, 840))

    # Hiển thị ảnh với kết quả nhận diện
    cv2.imshow("Face Recognition", img_display)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Đường dẫn ảnh cần nhận diện
image_path = "Face_Recognition/9ff11566-ef30-4938-8a2d-16243b19ab31.jpg"# Thay thế đường dẫn này

# Chạy nhận diện
recognize_face(image_path)
