import cv2
import numpy as np
import os
from matplotlib import pyplot as plt

imagg = cv2.imread(r"C:\Users\lokha\OneDrive\Pictures\robocondataset\fake_r2\1.png") # blue value [28,31 136, 255]
#img = cv2.imread(r"C:\Users\lokha\OneDrive\Pictures\robocondataset\real_ r2\2.png") # redvalue [221, 7, 21, 255]

def show_rgba_values(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        pixel = param[y, x]
        print(f"Clicked pixel at ({x},{y}) -> BGR: {pixel}")

image = cv2.imread(r"C:\Users\lokha\OneDrive\Pictures\robocondataset\real_ r2\2.png")
cv2.imshow("Image", image)
cv2.setMouseCallback("Image", show_rgba_values, image)
cv2.waitKey(0)
cv2.destroyAllWindows()
