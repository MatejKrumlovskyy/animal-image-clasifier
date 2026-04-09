Animal Image Classifier — CNN

A deep learning project for multi-class image classification using Convolutional Neural Networks (CNN). The model classifies images into 4 animal categories: **cow, cat, sheep, and spider**.

---

Project Overview

| Property | Details |
|---|---|
| Task | Multi-class image classification |
| Classes | Cow , Cat , Sheep , Spider |
| Dataset size | ~7,200 training images, ~1,240 test images |
| Best accuracy | **~76%** |
| Framework | TensorFlow / Keras |

---

Model Architecture

- `Conv2D` + `MaxPooling2D` layers for feature extraction
- `Dropout` layer to reduce overfitting
- `Dense` output layer with softmax activation
- Input image size: **64×64 RGB**

---

Key Techniques

- **Data Augmentation** — zoom, horizontal flip, shear (via `ImageDataGenerator`)
- **Dropout regularization** — prevents overfitting
- **Architecture comparison** — tested multiple layer configurations
- **Evaluation** — confusion matrix, classification report (precision, recall, F1)

---

Results

| Model variant | Test accuracy |
|---|---|
| Baseline CNN | ~72–73% |
| + Augmentation | ~74% |
| + Dropout (best) | **~76%** |

---

Author

**Matej Krumlovský** — FEI STU Bratislava  
[matejkrumlovsky8@gmail.com](mailto:matejkrumlovsky8@gmail.com)
