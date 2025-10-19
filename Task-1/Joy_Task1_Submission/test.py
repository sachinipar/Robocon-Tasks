import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sn
import os
from sklearn.metrics import classification_report, confusion_matrix

# --- 1. PARAMETERS ---
VALIDATION_DIR = os.path.join('dataset', 'validation')
IMAGE_WIDTH = 128
IMAGE_HEIGHT = 128
BATCH_SIZE = 32
MODEL_PATH = 'symbol_classifier_3class.keras' 

# --- 2. LOAD THE MODEL AND DATA ---
print(f"Loading model from {MODEL_PATH}...")
try:
    model = tf.keras.models.load_model(MODEL_PATH)
    print("Model loaded successfully.")
except Exception as e:
    print(f"Error loading model: {e}")
    exit()

print("Loading test data...")
test_dataset = tf.keras.utils.image_dataset_from_directory(
    VALIDATION_DIR,
    labels='inferred',
    label_mode='int',
    image_size=(IMAGE_HEIGHT, IMAGE_WIDTH),
    batch_size=BATCH_SIZE,
    shuffle=False
)

class_names = test_dataset.class_names
print(f"Classes found: {class_names}")

# --- 3. EVALUATE ---
print("\nEvaluating model on test data...")
y_true = np.concatenate([y for x, y in test_dataset], axis=0)
predictions_prob = model.predict(test_dataset)
y_pred = np.argmax(predictions_prob, axis=1)

# --- 4. DISPLAY RESULTS ---
print("\n" + "="*30)
print("--- 3-Class Classification Report ---")
print("="*30)
print(classification_report(y_true, y_pred, target_names=class_names, zero_division=0))

print("\nGenerating 3-Class confusion matrix...")
cm = confusion_matrix(y_true, y_pred)

plt.figure(figsize=(8, 6))
sn.heatmap(cm, annot=True, fmt='d', cmap='Blues',
           xticklabels=class_names, yticklabels=class_names)
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('3-Class Confusion Matrix')
plt.savefig('confusion_matrix_3class.png')

print("--- Confusion matrix saved to confusion_matrix_3class.png ---")
print("\nEvaluation complete.")

