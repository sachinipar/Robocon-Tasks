import tensorflow as tf
import os
import matplotlib.pyplot as plt

# --- 1. DEFINE YOUR PARAMETERS ---
DATASET_PATH = 'dataset'
TRAIN_DIR = os.path.join(DATASET_PATH, 'train')
VALIDATION_DIR = os.path.join(DATASET_PATH, 'validation')

IMAGE_WIDTH = 128
IMAGE_HEIGHT = 128
BATCH_SIZE = 32
EPOCHS = 40

# --- 2. LOAD THE DATA ---
print("Loading training data...")
train_dataset = tf.keras.utils.image_dataset_from_directory(
    TRAIN_DIR,
    labels='inferred',
    label_mode='int',
    image_size=(IMAGE_HEIGHT, IMAGE_WIDTH),
    batch_size=BATCH_SIZE,
    shuffle=True
)

print("Loading validation data...")
validation_dataset = tf.keras.utils.image_dataset_from_directory(
    VALIDATION_DIR,
    labels='inferred',
    label_mode='int',
    image_size=(IMAGE_HEIGHT, IMAGE_WIDTH),
    batch_size=BATCH_SIZE,
    shuffle=False
)

class_names = train_dataset.class_names
print(f"Classes found: {class_names}") 

# --- 3. BUILD THE SIMPLE CNN MODEL ---
model = tf.keras.models.Sequential([
    tf.keras.layers.Input(shape=(IMAGE_HEIGHT, IMAGE_WIDTH, 3)),
    tf.keras.layers.Rescaling(1./255),
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(len(class_names), activation='softmax')
])

print("\n--- Model Architecture ---")
model.summary()

# --- 4. COMPILE AND TRAIN ---
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

print("\n--- Starting Model Training ---")
history = model.fit(
    train_dataset,
    epochs=EPOCHS,
    validation_data=validation_dataset
)
print("--- Model Training Complete ---\n")

# --- 5. SAVE MODEL AND PLOT RESULTS ---
model.save('symbol_classifier_3class.keras')
print("--- Model saved to symbol_classifier_3class.keras ---\n")

# Visualization
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs_range = range(EPOCHS)

plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('3-Class Training and Validation Accuracy')

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('3-Class Training and Validation Loss')

plt.savefig('training_results_3class.png')
print("--- Training history plot saved to training_results_3class.png ---")

