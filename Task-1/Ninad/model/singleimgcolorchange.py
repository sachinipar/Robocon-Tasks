import cv2
import numpy as np
import os
# output= "C:\Users\lokha\OneDrive\Pictures\robocondataset\R2_fake"
# input = "C:\Users\lokha\OneDrive\Pictures\robocondataset\fake_r2\2.png"
# os.makedirs(output, exist_ok=True)
image = cv2.imread(r"C:\Users\lokha\OneDrive\Pictures\robocondataset\fake_r2\2.png")

target_red = np.array([136, 31, 28], dtype=np.int16)  # target RGB
tolerance = 100  # distance threshold

# Compute per-pixel distance from target red
diff = image.astype(np.int16) - target_red
dist = np.sqrt(np.sum(diff**2, axis=2))

# Mask pixels within tolerance
mask = dist < tolerance

# Apply blue
result = image.copy()
result[mask] = [21, 7, 221]  # BGR blue


# Show side by side
combined = np.hstack((image, result))
cv2.imshow("Original | Changed ", combined)
cv2.imwrite(r"C:\Users\lokha\OneDrive\Pictures\robocondataset\color_change_rgb_distance.jpg", combined)
cv2.waitKey(0)
cv2.destroyAllWindows()
