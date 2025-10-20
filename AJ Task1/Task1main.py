import tensorflow as tf
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import os

# ----------------- CONFIG -----------------
IMAGE_SIZE = (128, 128)
BATCH_SIZE = 8
SEED = 42
EPOCHS = 20
DATA_DIR = 'dataset'

print("Loading and preprocessing data...")

# ----------------- AUGMENTATION & NORMALIZATION -----------------
data_augmentation = tf.keras.Sequential([
    layers.RandomRotation(0.1),
    layers.RandomZoom(0.1),
], name='data_augmentation')

normalization_layer = layers.Rescaling(1. / 255)

# ----------------- DATASETS -----------------
train_dataset = tf.keras.utils.image_dataset_from_directory(
    os.path.join(DATA_DIR, 'train'),
    label_mode='categorical',
    seed=SEED,
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE
)

validation_dataset = tf.keras.utils.image_dataset_from_directory(
    os.path.join(DATA_DIR, 'validation'),
    label_mode='categorical',
    seed=SEED,
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=False
)

class_names = train_dataset.class_names
NUM_CLASSES = len(class_names)
print(f"Found classes: {class_names}")

# ----------------- PIPELINE OPTIMIZATION -----------------
def prepare(ds, augment=False):
    ds = ds.map(lambda x, y: (normalization_layer(x), y),
                num_parallel_calls=tf.data.AUTOTUNE)
    if augment:
        ds = ds.map(lambda x, y: (data_augmentation(x, training=True), y),
                    num_parallel_calls=tf.data.AUTOTUNE)
    return ds.cache().prefetch(buffer_size=tf.data.AUTOTUNE)

train_ds = prepare(train_dataset, augment=True)
val_ds = prepare(validation_dataset)

# ----------------- MODEL -----------------
print("Building the CNN model...")
IMG_SHAPE = IMAGE_SIZE + (3,)

model = models.Sequential([
    layers.Input(shape=IMG_SHAPE),

    # --- Feature Extraction Block 1 ---
    layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
    layers.MaxPooling2D((2, 2)),

    # --- Feature Extraction Block 2 ---
    layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
    layers.MaxPooling2D((2, 2)),

    # --- Feature Extraction Block 3 ---
    layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
    layers.MaxPooling2D((2, 2)),

    # --- Global Pooling ensures fixed output size ---
    layers.GlobalAveragePooling2D(),

    layers.Dense(128, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(NUM_CLASSES, activation='softmax')
])

# Compile
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Summary
model.summary()

# ----------------- TRAINING -----------------
print("\nStarting model training...")
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS
)
print("Model training complete.")

# ----------------- EVALUATION -----------------
print("\nEvaluating model performance...")

acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.plot(range(EPOCHS), acc, label='Training Accuracy')
plt.plot(range(EPOCHS), val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(1, 2, 2)
plt.plot(range(EPOCHS), loss, label='Training Loss')
plt.plot(range(EPOCHS), val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.tight_layout()
plt.savefig('training_history.png')
print("Saved training history plot to 'training_history.png'")

# ----------------- PREDICTION & REPORTS -----------------
print("Generating classification report and confusion matrix...")

# Predictions
y_pred_probabilities = model.predict(val_ds)
y_pred = np.argmax(y_pred_probabilities, axis=1)

# True labels
y_true = np.concatenate([y for _, y in val_ds], axis=0)
y_true = np.argmax(y_true, axis=1)

# Classification report
print("\n--- Classification Report ---")
print(classification_report(y_true, y_pred, target_names=class_names))

# Confusion matrix
cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=class_names, yticklabels=class_names)
plt.title('Confusion Matrix')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.tight_layout()
plt.savefig('confusion_matrix.png')
print("Saved confusion matrix plot to 'confusion_matrix.png'")

# ----------------- SAVE MODEL -----------------
model.save('robocon_symbol_classifier.h5')
print("Model saved as 'robocon_symbol_classifier.h5'")
