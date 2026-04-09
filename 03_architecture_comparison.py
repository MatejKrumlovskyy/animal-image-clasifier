"""
Animal Image Classifier - Architecture Comparison
==================================================
Systematically trains and compares 3 CNN architectures:
  - Model_Base:         filters 32/64,   dense 128, 5 epochs
  - Model_Enhanced:     filters 64/128,  dense 256, 7 epochs
  - Model_With_Dropout: filters 32/64,   dense 128, 10 epochs + Dropout(0.5)

Results (accuracy, training time, confusion matrix) are printed at the end.

Author: Matej Krumlovsky
"""

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import time

# Data Preprocessing

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

# Training Function

def build_and_train_model(filters1, filters2, dense_units, epochs, model_name, use_dropout=False):
    """Build, train and evaluate a CNN model. Returns a results dict."""
    print(f"\n{'='*50}")
    print(f"Training: {model_name}")
    print(f"{'='*50}")
    start_time = time.time()

    model = Sequential([
        Conv2D(filters=filters1, kernel_size=3, activation='relu', input_shape=[64, 64, 3]),
        MaxPooling2D(pool_size=2, strides=2),
        Conv2D(filters=filters2, kernel_size=3, activation='relu'),
        MaxPooling2D(pool_size=2, strides=2),
        Flatten(),
        Dense(units=dense_units, activation='relu'),
        *([ Dropout(0.5) ] if use_dropout else []),
        Dense(units=4, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    model.summary()

    history = model.fit(x=training_set, validation_data=test_set, epochs=epochs)
    training_time = time.time() - start_time

    # Evaluation
    eval_results = model.evaluate(test_set, batch_size=32, verbose=0)
    print(f"\n{model_name} → Test loss: {eval_results[0]:.4f} | Test accuracy: {eval_results[1]:.4f}")

    test_set.reset()
    predictions = model.predict(test_set)
    predicted_classes = np.argmax(predictions, axis=1)
    true_classes = test_set.classes
    accuracy = accuracy_score(true_classes, predicted_classes)
    cm = confusion_matrix(true_classes, predicted_classes)

    # Training curves
    plt.figure(figsize=(14, 5))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.legend(loc='lower right')
    plt.title(f"{model_name} — Accuracy")

    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.legend(loc='upper right')
    plt.title(f"{model_name} — Loss")
    plt.tight_layout()
    plt.savefig(f'results/{model_name}_training_curves.png')
    plt.show()

    # Confusion matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=test_set.class_indices.keys(),
                yticklabels=test_set.class_indices.keys())
    plt.title(f"{model_name} — Confusion Matrix")
    plt.xlabel("Predicted Labels")
    plt.ylabel("True Labels")
    plt.tight_layout()
    plt.savefig(f'results/{model_name}_confusion_matrix.png')
    plt.show()

    print(f"\n{model_name} — Classification Report:")
    print(classification_report(true_classes, predicted_classes,
                                 target_names=test_set.class_indices.keys()))

    return {
        "model_name": model_name,
        "accuracy": accuracy,
        "training_time": training_time,
        "confusion_matrix": cm
    }

# Experiments

results = []

results.append(build_and_train_model(
    filters1=32, filters2=64, dense_units=128,
    epochs=5, model_name="Model_Base"
))

results.append(build_and_train_model(
    filters1=64, filters2=128, dense_units=256,
    epochs=7, model_name="Model_Enhanced"
))

results.append(build_and_train_model(
    filters1=32, filters2=64, dense_units=128,
    epochs=10, model_name="Model_With_Dropout",
    use_dropout=True
))

# Summary

print("\n" + "="*50)
print("RESULTS SUMMARY")
print("="*50)
for result in results:
    print(f"\nModel:          {result['model_name']}")
    print(f"Accuracy:       {result['accuracy']:.4f} ({result['accuracy']*100:.2f}%)")
    print(f"Training time:  {result['training_time']:.2f} seconds")
    print(f"Confusion Matrix:\n{result['confusion_matrix']}")
    print("-"*50)
