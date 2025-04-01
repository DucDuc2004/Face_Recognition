import os
import json

# Đường dẫn đến thư mục dataset (chứa các thư mục ID_Ho_Ten)
dataset_path = "Face_Recognition/data"  # Thay đường dẫn thực tế

# Tạo danh sách nhãn tự động
label_dict = {}
label_id = 0

for folder in os.listdir(dataset_path):
    folder_path = os.path.join(dataset_path, folder)
    if os.path.isdir(folder_path):  # Kiểm tra có phải thư mục không
        if folder not in label_dict:
            label_dict[folder] = label_id
            label_id += 1

# Lưu danh sách nhãn vào file JSON
with open("label_map.json", "w") as f:
    json.dump(label_dict, f)

print("Đã tạo nhãn và lưu vào label_map.json")
