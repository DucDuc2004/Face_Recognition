import cv2
import os
import threading

cam = cv2.VideoCapture(0)

cam.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
cam.set(cv2.CAP_PROP_FPS, 30) 

face_detect = cv2.CascadeClassifier("Face_Recognition/model/haarcascade_frontalface_default.xml")

name = input("Họ và Tên: ")
id = input("ID: ")

data = "Face_Recognition/data"
folder = os.path.join(data, f"{id}_{name.replace(' ', '_')}")
if not os.path.exists(folder):
    os.makedirs(folder)

image_number = 0
num_images = 200

def capture_images():
    global image_number
    print("\nNhìn vào Camera")
    
    while image_number < num_images:
        ret, img = cam.read()
        if not ret:
            continue
        
        img = cv2.flip(img, 1)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_detect.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

        for (x, y, w, h) in faces:
            if w * h > 10000:
                image_number += 1
                face_img = gray[y:y+h, x:x+w]
                face_img_resized = cv2.resize(face_img, (200, 200))

                filename = f"User.{id}.{image_number}.jpg"
                path = os.path.join(folder, filename)
                cv2.imwrite(path, face_img_resized)

                cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
                cv2.putText(img, f"Image: {image_number}/{num_images}", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        cv2.imshow('image', img)
        if cv2.waitKey(1) & 0xFF == 27: 
            break

thread = threading.Thread(target=capture_images)
thread.start()
thread.join()

print("Thu thập dữ liệu hoàn tất.")
cam.release()
cv2.destroyAllWindows()
