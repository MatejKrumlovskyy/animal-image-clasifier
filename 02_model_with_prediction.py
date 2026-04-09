# -*- coding: utf-8 -*-
"""
Animal Image Classifier - Model with Single Image Prediction
=============================================================
Extends the baseline model with the ability to predict a single image
and overlay the predicted class label onto the image using OpenCV.

Author: Matej Krumlovsky
"""

import tensorflow as tf
from tensorflow.keras.utils import load_img, img_to_array
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import cv2

# ── Helper Functions ──────────────────────────────────────────────────────────

CLASS_LABELS = {0: "krava", 1: "macka", 2: "ovca", 3: "pavuk"}

def predict_single_image(model, image_path):
    """Load an image, preprocess it and return model prediction."""
    img = load_img(image_path, target_size=(64, 64))
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0
    result = model.predict(img_array)
    predicted_class = np.argmax(result, axis=1)[0]
    print(f"Prediction: {CLASS_LABELS[predicted_class]} (confidence: {result[0][predicted_class]:.2%})")
    return predicted_class

def save_prediction_overlay(image_path, predicted_class, output_name):
    """Write predicted class label onto image and save it."""
    img = cv2.imread(image_path)
    label = CLASS_LABELS[predicted_class]
    cv2.putText(img, label, (10, 50), cv2.FONT_HERSHEY_SIMPLEX,
                1, (0, 255, 0), 2, cv2.LINE_AA)
    output_path = f"results/predicted_{output_name}.jpg"
    cv2.imwrite(output_path, img)
    print(f"Saved: {output_path}")

# ── Data Preprocessing ────────────────────────────────────────────────────────

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
    class_mode='categorical'
)

test_datagen = ImageDataGenerator(rescale=1./255)
test_set = test_datagen.flow_from_directory(
    'datasetZ2/TEST',
    target_size=(64, 64),
    batch_size=32,
    class_mode='categorical',
    shuffle=False
)

# ── Model Architecture ────────────────────────────────────────────────────────

cnn = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(filters=32, kernel_size=3, activation='relu', input_shape=[64, 64, 3]),
    tf.keras.layers.MaxPool2D(pool_size=2, strides=2),
    tf.keras.layers.Conv2D(filters=32, kernel_size=3, activation='relu'),
    tf.keras.layers.MaxPool2D(pool_size=2, strides=2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(units=128, activation='relu'),
    tf.keras.layers.Dense(units=4, activation='softmax')
])

cnn.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
cnn.summary()

# ── Training ──────────────────────────────────────────────────────────────────

history = cnn.fit(x=training_set, validation_data=test_set, epochs=15)

# ── Evaluation ────────────────────────────────────────────────────────────────

results_validation = cnn.evaluate(test_set, batch_size=32)
print(f"\nTest loss: {results_validation[0]:.4f} | Test accuracy: {results_validation[1]:.4f}")

test_set.reset()
predictions = cnn.predict(test_set, batch_size=32)
predicted_classes = np.argmax(predictions, axis=1)

accuracy = accuracy_score(test_set.classes, predicted_classes)
print(f"Accuracy score: {accuracy:.4f}")

# ── Confusion Matrix ──────────────────────────────────────────────────────────

cm = confusion_matrix(test_set.classes, predicted_classes)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=test_set.class_indices.keys(),
            yticklabels=test_set.class_indices.keys())
plt.title("Confusion Matrix")
plt.xlabel("Predicted Labels")
plt.ylabel("True Labels")
plt.tight_layout()
plt.savefig('results/prediction_confusion_matrix.png')
plt.show()

print("\nClassification Report:")
print(classification_report(test_set.classes, predicted_classes,
                             target_names=test_set.class_indices.keys()))

# ── Single Image Prediction Example ──────────────────────────────────────────
# Uncomment and set your image path to test on a single image:

# predicted = predict_single_image(cnn, 'path/to/your/image.jpg')
# save_prediction_overlay('path/to/your/image.jpg', predicted, 'test_output')
