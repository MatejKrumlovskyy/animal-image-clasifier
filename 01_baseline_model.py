# -*- coding: utf-8 -*-
"""
Animal Image Classifier - Baseline CNN Model
=============================================
A simple CNN trained to classify 4 animal categories:
  - Cow (krava)
  - Cat (macka)
  - Sheep (ovca)
  - Spider (pavuk)

Dataset structure expected:
  datasetZ2/
    TRAINING/
      cow/ cat/ sheep/ spider/
    TEST/
      cow/ cat/ sheep/ spider/

Author: Matej Krumlovsky
"""

import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# ── Data Preprocessing ──────────────────────────────────────────────────────

train_datagen = ImageDataGenerator(
    rescale=1./255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True
)
training_set = train_datagen.flow_from_directory(
    'datasetZ2/TRAINING',
    target_size=(64, 64),
    batch_size=32,
    class_mode='categorical',
    shuffle=True
)

test_datagen = ImageDataGenerator(rescale=1./255)
test_set = test_datagen.flow_from_directory(
    'datasetZ2/TEST',
    target_size=(64, 64),
    batch_size=32,
    class_mode='categorical',
    shuffle=False
)

# ── Model Architecture ───────────────────────────────────────────────────────

cnn = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(filters=32, kernel_size=3, activation='relu', input_shape=[64, 64, 3]),
    tf.keras.layers.MaxPool2D(pool_size=2, strides=2),
    tf.keras.layers.Conv2D(filters=64, kernel_size=3, activation='relu'),
    tf.keras.layers.MaxPool2D(pool_size=2, strides=2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(units=128, activation='relu'),
    tf.keras.layers.Dense(units=4, activation='softmax')
])

cnn.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
cnn.summary()

# ── Training ─────────────────────────────────────────────────────────────────

history = cnn.fit(x=training_set, validation_data=test_set, epochs=5)

# ── Evaluation ───────────────────────────────────────────────────────────────

results_validation = cnn.evaluate(test_set, batch_size=32)
print(f"\nTest loss: {results_validation[0]:.4f} | Test accuracy: {results_validation[1]:.4f}")

test_set.reset()
predictions = cnn.predict(test_set)
predicted_classes = np.argmax(predictions, axis=1)

accuracy = accuracy_score(test_set.classes, predicted_classes)
print(f"Accuracy score: {accuracy:.4f}")

# ── Confusion Matrix ──────────────────────────────────────────────────────────

cm = confusion_matrix(test_set.classes, predicted_classes)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=test_set.class_indices.keys(),
            yticklabels=test_set.class_indices.keys())
plt.title("Baseline Model — Confusion Matrix")
plt.xlabel("Predicted Labels")
plt.ylabel("True Labels")
plt.tight_layout()
plt.savefig('results/baseline_confusion_matrix.png')
plt.show()

# ── Classification Report ────────────────────────────────────────────────────

print("\nClassification Report:")
print(classification_report(test_set.classes, predicted_classes,
                             target_names=test_set.class_indices.keys()))

# ── Training Curves ──────────────────────────────────────────────────────────

plt.figure(figsize=(14, 5))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Baseline Model — Accuracy')

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Baseline Model — Loss')
plt.tight_layout()
plt.savefig('results/baseline_training_curves.png')
plt.show()
