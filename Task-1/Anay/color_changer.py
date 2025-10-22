import cv2
import numpy as np
import os

input_folder = r"file path"
output_folder = r"file path"
os.makedirs(output_folder, exist_ok=True)

red_bgr = np.array([22, 7, 222])    # rgba(222, 7, 22)
blue_bgr = np.array([131, 34, 28])  # rgba(28, 34, 131)

tolerance = 90 # change as per need (increase if you see some color left)

reverse_mode = True

source_color = blue_bgr if reverse_mode else red_bgr
target_color = red_bgr if reverse_mode else blue_bgr

suffix = "_red" if reverse_mode else "_blue"

for filename in os.listdir(input_folder):
    if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
        path = os.path.join(input_folder, filename)
        img = cv2.imread(path)

        diff = np.abs(img - source_color)
        mask = np.all(diff <= tolerance, axis=-1)

        img[mask] = target_color

        name, ext = os.path.splitext(filename)
        new_filename = f"{name}{suffix}{ext}"
        output_path = os.path.join(output_folder, new_filename)

        cv2.imwrite(output_path, img)
        print(f"Processed: {new_filename}")

print("renamed and saved")
