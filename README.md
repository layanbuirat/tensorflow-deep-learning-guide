# TensorFlow Deep Learning Guide

[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.16.1-FF6F00?style=for-the-badge&logo=tensorflow)](https://tensorflow.org)
[![Python](https://img.shields.io/badge/Python-3.8+-3776AB?style=for-the-badge&logo=python)](https://python.org)
[![Keras](https://img.shields.io/badge/Keras-3.0+-D00000?style=for-the-badge&logo=keras)](https://keras.io)
[![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)](LICENSE)

A complete guide to building, training, and deploying deep learning models using TensorFlow 2.x and Keras. From tensors to transfer learning - everything you need to master neural networks.

---

## 📋 Table of Contents

- [Overview](#overview)
- [What You'll Learn](#what-youll-learn)
- [Installation](#installation)
- [Tensors - The Building Blocks](#tensors---the-building-blocks)
- [TensorFlow Datasets](#tensorflow-datasets)
- [Data Pipeline](#data-pipeline)
- [Building Neural Networks](#building-neural-networks)
- [Model Architecture](#model-architecture)
- [Compiling the Model](#compiling-the-model)
- [Training the Model](#training-the-model)
- [Loss Functions Guide](#loss-functions-guide)
- [Preventing Overfitting](#preventing-overfitting)
- [Callbacks](#callbacks)
- [Saving and Loading Models](#saving-and-loading-models)
- [Transfer Learning](#transfer-learning)
- [GPU Acceleration](#gpu-acceleration)
- [Common Questions & Answers](#common-questions--answers)
- [Complete Code Examples](#complete-code-examples)
- [Resources](#resources)
- [License](#license)

---

## 🎯 Overview

This repository provides a comprehensive learning path for mastering TensorFlow 2.x and Keras for deep learning applications. From the fundamental concept of **tensors** to advanced **transfer learning** techniques, you'll build practical skills through hands-on exercises with real datasets.

### What You Will Achieve

- ✅ Classify handwritten digits with 98%+ accuracy
- ✅ Build multi-layer neural networks from scratch
- ✅ Implement early stopping and dropout for robust models
- ✅ Save and load models in multiple formats (Keras v3, HDF5, SavedModel)
- ✅ Apply transfer learning for faster training
- ✅ Use GPU acceleration for large-scale training

---

## 📚 What You'll Learn

| Topic | Skills Gained |
|-------|---------------|
| **Tensors** | Create, manipulate, and convert tensors to/from NumPy |
| **Data Pipelines** | Build efficient input pipelines with tf.data |
| **Neural Networks** | Design sequential models with dense layers |
| **Activation Functions** | Use ReLU, Softmax, Sigmoid, Tanh |
| **Loss Functions** | Apply MSE, Categorical Crossentropy, Binary Crossentropy |
| **Optimizers** | Configure Adam, SGD, RMSprop |
| **Regularization** | Implement Dropout to prevent overfitting |
| **Callbacks** | Use EarlyStopping and ModelCheckpoint |
| **Model Saving** | Save in Keras v3, HDF5, and SavedModel formats |
| **Transfer Learning** | Adapt pre-trained networks for custom tasks |
| **GPU Computing** | Leverage GPU acceleration for faster training |

---

## 🔧 Installation

### Option 1: Using Conda (Recommended)

```bash
conda create -n tf2-env python=3.9
conda activate tf2-env
conda install numpy jupyter notebook
pip install tensorflow==2.16.1 tensorflow-datasets tensorflow-hub
```

### Option 2: Using pip

```bash
pip install tensorflow==2.16.1 tensorflow-datasets tensorflow-hub numpy jupyter
```

### Option 3: Google Colab (No Setup Required)

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/)

---

## 🧮 Tensors - The Building Blocks

Tensors are the fundamental data structure in TensorFlow, similar to NumPy arrays but with GPU support.

```python
import tensorflow as tf
import numpy as np

# Create tensors from lists
tensor_1d = tf.constant([1, 2, 3, 4])
tensor_2d = tf.constant([[1, 2], [3, 4]])
tensor_3d = tf.constant([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])

# Convert from NumPy
numpy_array = np.array([1, 2, 3, 4])
tensor_from_numpy = tf.convert_to_tensor(numpy_array)

# Convert back to NumPy
numpy_again = tensor_from_numpy.numpy()

# Check shape and dtype
print(f"Shape: {tensor_2d.shape}")
print(f"Data type: {tensor_2d.dtype}")
```

**Key Points:**
- All arrays qualify as tensors (1D, 2D, 3D+, n-dimensional)
- Tensors and NumPy arrays have independent values
- Use `.numpy()` to convert tensor to NumPy array

---

## 📦 TensorFlow Datasets

TensorFlow Datasets (tfds) provides a collection of ready-to-use datasets.

```python
import tensorflow_datasets as tfds

# Load Fashion-MNIST dataset
dataset, info = tfds.load('fashion_mnist', as_supervised=True, with_info=True)

# Access splits
training_set = dataset['train']
test_set = dataset['test']

# Get dataset information
print(f"Dataset: {info.name}")
print(f"Version: {info.version}")
print(f"Training examples: {info.splits['train'].num_examples}")
print(f"Test examples: {info.splits['test'].num_examples}")
print(f"Classes: {info.features['label'].num_classes}")
```

### Available Datasets in TFDS

- `mnist` - Handwritten digits (60k train, 10k test)
- `fashion_mnist` - Clothing items (60k train, 10k test)
- `cifar10` / `cifar100` - Color images (50k train, 10k test)
- `kitti` - Autonomous driving data
- `titanic` - Passenger survival data
- `newsroom` - News articles

---

## 🔄 Data Pipeline

Building efficient data pipelines is crucial for training performance.

```python
def normalize(image, label):
    """Normalize images to range [0, 1]"""
    image = tf.cast(image, tf.float32)
    image = image / 255.0
    return image, label

# Create training pipeline
batch_size = 64

training_batches = training_set.cache() \
    .shuffle(60000) \
    .batch(batch_size) \
    .map(normalize) \
    .prefetch(1)

testing_batches = test_set.cache() \
    .batch(batch_size) \
    .map(normalize) \
    .prefetch(1)
```

### Pipeline Methods Explained

| Method | Purpose |
|--------|---------|
| `.cache()` | Stores dataset in memory for faster access |
| `.shuffle()` | Randomizes order to avoid local minima |
| `.batch()` | Groups data into batches for efficient processing |
| `.map()` | Applies preprocessing function to each element |
| `.prefetch()` | Overlaps preprocessing and model execution |

### Creating Train/Validation/Test Splits

```python
# Method 1: Using percentages
training_set, validation_set, test_set = tfds.load(
    'fashion_mnist',
    split=['train[:80%]', 'train[80%:90%]', 'train[90%:]'],
    as_supervised=True
)

# Method 2: Using exact counts
training_set = tfds.load('fashion_mnist', split='train[:48000]', as_supervised=True)
validation_set = tfds.load('fashion_mnist', split='train[48000:60000]', as_supervised=True)
test_set = tfds.load('fashion_mnist', split='test', as_supervised=True)
```

---

## 🏗️ Building Neural Networks

### Basic Sequential Model

```python
model = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28, 1)),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax')
])

model.summary()
```

### Model with Dropout (Prevents Overfitting)

```python
model = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28, 1)),
    tf.keras.layers.Dense(256, activation='relu'),
    tf.keras.layers.Dropout(0.2),  # Drops 20% of neurons
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(10, activation='softmax')
])
```

### Custom Model Using Subclassing

```python
class MyCustomModel(tf.keras.Model):
    def __init__(self):
        super(MyCustomModel, self).__init__()
        self.flatten = tf.keras.layers.Flatten(input_shape=(28, 28, 1))
        self.dense1 = tf.keras.layers.Dense(128, activation='relu')
        self.dense2 = tf.keras.layers.Dense(64, activation='relu')
        self.dense3 = tf.keras.layers.Dense(10, activation='softmax')
    
    def call(self, x):
        x = self.flatten(x)
        x = self.dense1(x)
        x = self.dense2(x)
        x = self.dense3(x)
        return x

model = MyCustomModel()
```

### Weight Matrix Shapes

For a network with: **10 inputs → 5 hidden → 2 outputs**

```python
# Valid shapes
W1 = (10, 5)   # Weights from input to hidden layer
B1 = (1, 5)    # Biases for hidden layer
W2 = (5, 2)    # Weights from hidden to output layer
B2 = (1, 2)    # Biases for output layer

# Invalid shapes (won't work)
# W1: (10, 2) - Wrong dimension
# B2: (2, 1) - Wrong shape (should be 1x2)
```

---

## ⚙️ Compiling the Model

```python
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)
```

### Available Optimizers

| Optimizer | Best For |
|-----------|----------|
| `'adam'` | General purpose (recommended) |
| `'sgd'` | Simple problems, more control |
| `'rmsprop'` | Recurrent neural networks |
| `'adagrad'` | Sparse data |

---

## 📉 Loss Functions Guide

| Task | Loss Function | When to Use |
|------|---------------|-------------|
| Binary Classification | `'binary_crossentropy'` | Two classes (cat/dog, spam/not spam) |
| Multi-class Classification | `'sparse_categorical_crossentropy'` | Labels are integers (0, 1, 2, ...) |
| Multi-class Classification | `'categorical_crossentropy'` | Labels are one-hot encoded |
| Regression (price prediction) | `'mean_squared_error'` | Predicting continuous values |
| Regression (robust) | `'mean_absolute_error'` | When outliers are present |
| Similarity Tasks | `'cosine_similarity'` | Comparing embeddings |

### Example: Choosing loss function for home price prediction

```python
# For predicting house prices (regression task)
model.compile(
    optimizer='adam',
    loss='mean_squared_error',  # NOT sparse_categorical_crossentropy
    metrics=['mae']
)
```

---

## 🚀 Training the Model

```python
# Basic training
history = model.fit(
    training_batches,
    epochs=5
)

# Training with validation
history = model.fit(
    training_batches,
    epochs=10,
    validation_data=validation_batches
)

# Training with callbacks
history = model.fit(
    training_batches,
    epochs=100,
    validation_data=validation_batches,
    callbacks=[early_stopping, model_checkpoint]
)
```

### Understanding the Output

```
Epoch 1/5
938/938 [==============================] - 23s 24ms/step - loss: 0.2768 - accuracy: 0.9204
Epoch 2/5
938/938 [==============================] - 20s 21ms/step - loss: 0.1205 - accuracy: 0.9651
```

- **loss**: Training loss (lower is better)
- **accuracy**: Training accuracy (higher is better)
- **val_loss**: Validation loss (if validation_data provided)
- **val_accuracy**: Validation accuracy (if validation_data provided)

### Making Predictions

```python
# Predict on a batch
for image_batch, label_batch in testing_batches.take(1):
    predictions = model.predict(image_batch)
    first_prediction = predictions[0]
    predicted_class = np.argmax(first_prediction)
    
print(f"Predicted class: {predicted_class}")
print(f"Class probabilities: {first_prediction}")
```

---

## 🛡️ Preventing Overfitting

### 1. Early Stopping

Stop training when validation loss stops improving.

```python
early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss',     # Monitor validation loss
    patience=5,             # Stop after 5 epochs without improvement
    min_delta=0.001,        # Minimum change to qualify as improvement
    restore_best_weights=True  # Restore best model weights
)

history = model.fit(
    training_batches,
    epochs=100,
    validation_data=validation_batches,
    callbacks=[early_stopping]
)
```

### 2. Dropout Layers

Randomly drop neurons during training to force redundant learning.

```python
model = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28, 1)),
    tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.Dropout(0.5),  # Drop 50% of neurons
    tf.keras.layers.Dense(256, activation='relu'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(10, activation='softmax')
])
```

**Important:** TensorFlow automatically disables dropout during inference/prediction. You don't need to remove them manually.

### 3. Validation Set

Always keep a separate validation set:

```python
# 80% train, 20% validation
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size

training_set = dataset.take(train_size)
validation_set = dataset.skip(train_size)
```

### Detecting Overfitting

| Training Loss | Validation Loss | Conclusion |
|---------------|-----------------|------------|
| Decreasing | Decreasing | ✅ Good training |
| Decreasing | Increasing | ⚠️ Overfitting! |
| Low | High | ⚠️ Severe overfitting |
| High | High | ❌ Underfitting |

---

## 📞 Callbacks

### ModelCheckpoint - Save Best Model

```python
# Save only the best model based on validation loss
model_checkpoint = tf.keras.callbacks.ModelCheckpoint(
    'best_model.keras',
    monitor='val_loss',
    save_best_only=True,
    verbose=1
)

# Save model every epoch
model_checkpoint_all = tf.keras.callbacks.ModelCheckpoint(
    'model_epoch_{epoch:02d}.keras',
    save_freq='epoch'
)
```

### EarlyStopping - Stop When No Improvement

```python
early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss',
    patience=5,
    min_delta=0.001,
    restore_best_weights=True
)
```

### ReduceLROnPlateau - Reduce Learning Rate

```python
reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.2,      # Multiply learning rate by 0.2
    patience=3,      # Wait 3 epochs before reducing
    min_lr=0.00001   # Minimum learning rate
)
```

### TensorBoard - Visualize Training

```python
tensorboard_callback = tf.keras.callbacks.TensorBoard(
    log_dir='./logs',
    histogram_freq=1
)

history = model.fit(
    training_batches,
    epochs=10,
    callbacks=[tensorboard_callback]
)

# Launch TensorBoard: tensorboard --logdir ./logs
```

---

## 💾 Saving and Loading Models

### Save Format Comparison

| Format | Extension | Best For | File Type |
|--------|-----------|----------|-----------|
| **Keras v3** | `.keras` | Easy debugging, future compatibility | Single zip file |
| **SavedModel** | Directory | Web browsers, edge devices, TF Serving | Directory |
| **HDF5** | `.h5` | Single file management, legacy systems | Single file |

### Saving Models

```python
# Keras v3 format (RECOMMENDED)
model.save('my_model.keras')

# SavedModel format
model.save('saved_model_directory')

# HDF5 format
model.save('my_model.h5')

# Save with timestamp
import time
t = time.time()
model.save(f'./model_{int(t)}.keras')
```

### Loading Models

```python
# Load any format
loaded_model = tf.keras.models.load_model('my_model.keras')

# Continue training
loaded_model.fit(training_batches, epochs=5)

# Make predictions
predictions = loaded_model.predict(testing_batches)
```

### What Gets Saved

The saved model contains:
- Model architecture (layers, connections)
- Model weights (learned parameters)
- Training configuration (optimizer, loss function)
- Optimizer state (for resuming training)

---

## 🔄 Transfer Learning

### Four Scenarios

| Data Size | Similarity to Original | Strategy |
|-----------|----------------------|----------|
| Small | Similar | Freeze base, add new head |
| Small | Different | Freeze lower layers only, add new head |
| Large | Similar | Fine-tune entire network |
| Large | Different | Train from scratch (or fine-tune) |

### Implementation Examples

```python
# Load pre-trained model (MobileNetV2)
base_model = tf.keras.applications.MobileNetV2(
    input_shape=(224, 224, 3),
    include_top=False,      # Remove classification head
    weights='imagenet'      # Use ImageNet pre-trained weights
)

# Case 1 & 2: Small dataset - Freeze base model
base_model.trainable = False

# Add custom classification head
model = tf.keras.Sequential([
    base_model,
    tf.keras.layers.GlobalAveragePooling2D(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(10, activation='softmax')  # 10 classes
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Case 3 & 4: Large dataset - Fine-tune
base_model.trainable = True  # Unfreeze for fine-tuning

# Use lower learning rate for fine-tuning
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001), ...)
```

### Pre-trained Models Available

| Model | Size | Accuracy (Top-5) |
|-------|------|------------------|
| MobileNetV2 | 14 MB | 90.5% |
| ResNet50 | 98 MB | 92.1% |
| VGG16 | 528 MB | 90.0% |
| InceptionV3 | 92 MB | 93.9% |
| EfficientNetB0 | 29 MB | 93.0% |

---

## 🖥️ GPU Acceleration

### Checking GPU Availability

```python
import tensorflow as tf

# List available GPUs
gpus = tf.config.list_physical_devices('GPU')
print(f"GPUs available: {gpus}")

# Check if GPU is being used
print(f"GPU Available: {bool(gpus)}")
print(f"TensorFlow Version: {tf.__version__}")

# Check GPU details
if gpus:
    for gpu in gpus:
        print(f"GPU: {gpu}")
```

### GPU Best Practices

```python
# 1. Use larger batch sizes for GPU
batch_size = 128  # Instead of 32 or 64

# 2. Use prefetch for data pipeline
training_batches = dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)

# 3. Mixed precision training (faster on modern GPUs)
from tensorflow.keras import mixed_precision
mixed_precision.set_global_policy('mixed_float16')

# 4. Limit GPU memory growth (prevents allocation errors)
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)
```

### GPU vs CPU Workflow

1. **Develop on CPU** - Quick iteration, save GPU hours
2. **Test on small subset** - Ensure network learns
3. **Train on GPU** - Full training when ready

---

## ❓ Common Questions & Answers

### Q1: What qualifies as tensors?
**A:** All arrays qualify as tensors - 1-dimensional, 2-dimensional, 3-dimensional, and n-dimensional arrays are all tensors.

### Q2: Does tf.matmul() transpose arguments modify the original matrix?
**A:** No. Setting `transpose_a=True` or `transpose_b=True` only transposes the matrices within the function call. The original matrices remain unchanged.

### Q3: Do I need to manually remove Dropout layers for testing?
**A:** No. TensorFlow automatically disables Dropout layers during inference and evaluation.

### Q4: Which Softmax functions are available in TensorFlow?
**A:** `tf.nn.softmax` and `tf.math.softmax`. `tf.layers.softmax` is deprecated.

### Q5: How do I know if my model is overfitting?
**A:** When validation loss increases while training loss decreases, or when validation accuracy is significantly lower than training accuracy.

### Q6: What happens if I run training multiple times without clearing the graph?
**A:** Layer names continue getting higher in number (dense_1, dense_2, dense_3...), but the model doesn't get larger.

### Q7: Do I need to calculate GradientTape manually for each network?
**A:** No. Keras handles automatic differentiation automatically. GradientTape is for custom training loops.

---

## 💻 Complete Code Examples

### Example 1: MNIST Classifier

```python
import tensorflow as tf
import tensorflow_datasets as tfds

# Load data
dataset, info = tfds.load('mnist', as_supervised=True, with_info=True)
train_set, test_set = dataset['train'], dataset['test']

# Preprocess
def preprocess(img, label):
    img = tf.cast(img, tf.float32) / 255.0
    return img, label

# Pipeline
train_batches = train_set.cache().shuffle(60000).batch(32).map(preprocess).prefetch(1)
test_batches = test_set.batch(32).map(preprocess)

# Build model
model = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28, 1)),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(10, activation='softmax')
])

# Compile
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train
history = model.fit(train_batches, epochs=5, validation_data=test_batches)

# Evaluate
loss, accuracy = model.evaluate(test_batches)
print(f"Test Accuracy: {accuracy:.3%}")
```

### Example 2: Fashion-MNIST with Early Stopping

```python
import tensorflow as tf
import tensorflow_datasets as tfds

# Load with splits
train_set, val_set, test_set = tfds.load(
    'fashion_mnist',
    split=['train[:80%]', 'train[80%:90%]', 'train[90%:]'],
    as_supervised=True
)

# Preprocess
def preprocess(img, label):
    img = tf.cast(img, tf.float32) / 255.0
    return img, label

# Pipelines
batch_size = 64
train_batches = train_set.cache().shuffle(48000).batch(batch_size).map(preprocess).prefetch(1)
val_batches = val_set.cache().batch(batch_size).map(preprocess).prefetch(1)
test_batches = test_set.batch(batch_size).map(preprocess)

# Build model
model = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28, 1)),
    tf.keras.layers.Dense(256, activation='relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(10, activation='softmax')
])

# Compile
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Callbacks
early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss', patience=5, restore_best_weights=True
)

checkpoint = tf.keras.callbacks.ModelCheckpoint(
    'best_fashion_model.keras', monitor='val_loss', save_best_only=True
)

# Train
history = model.fit(
    train_batches,
    epochs=50,
    validation_data=val_batches,
    callbacks=[early_stopping, checkpoint]
)

# Evaluate
loss, accuracy = model.evaluate(test_batches)
print(f"Test Accuracy: {accuracy:.3%}")
```

### Example 3: Custom Training Loop with GradientTape

```python
import tensorflow as tf

# Simple example: y = x^2
x = tf.random.normal((2, 2))

with tf.GradientTape() as tape:
    tape.watch(x)  # Watch this tensor
    y = x ** 2     # y = x^2, dy/dx = 2x

dy_dx = tape.gradient(y, x)
true_grad = 2 * x

print(f"Gradient from tf.GradientTape:\n{dy_dx}")
print(f"True gradient:\n{true_grad}")
print(f"Difference: {tf.reduce_max(tf.abs(true_grad - dy_dx))}")
```

### Example 4: Transfer Learning with MobileNetV2

```python
import tensorflow as tf

# Load pre-trained model
base_model = tf.keras.applications.MobileNetV2(
    input_shape=(224, 224, 3),
    include_top=False,
    weights='imagenet'
)

# Freeze base model
base_model.trainable = False

# Build complete model
model = tf.keras.Sequential([
    tf.keras.layers.Resizing(224, 224),  # Resize images to MobileNet input size
    tf.keras.layers.Rescaling(1./255),
    base_model,
    tf.keras.layers.GlobalAveragePooling2D(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(10, activation='softmax')
])

# Compile
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# Summary
model.summary()

# Train
history = model.fit(train_batches, epochs=10, validation_data=val_batches)
```

---

## 📊 Visualizing Training History

```python
import matplotlib.pyplot as plt

def plot_training_history(history):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    # Plot accuracy
    ax1.plot(history.history['accuracy'], label='Training Accuracy')
    if 'val_accuracy' in history.history:
        ax1.plot(history.history['val_accuracy'], label='Validation Accuracy')
    ax1.set_title('Model Accuracy')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Accuracy')
    ax1.legend()
    
    # Plot loss
    ax2.plot(history.history['loss'], label='Training Loss')
    if 'val_loss' in history.history:
        ax2.plot(history.history['val_loss'], label='Validation Loss')
    ax2.set_title('Model Loss')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss')
    ax2.legend()
    
    plt.tight_layout()
    plt.show()

# Use the function
plot_training_history(history)
```

---

## 🔍 Model Evaluation and Prediction Visualization

```python
import matplotlib.pyplot as plt
import numpy as np

def visualize_predictions(model, dataset, class_names, num_images=10):
    """Visualize model predictions on random images"""
    for image_batch, label_batch in dataset.take(1):
        predictions = model.predict(image_batch)
        images = image_batch.numpy().squeeze()
        labels = label_batch.numpy()
    
    plt.figure(figsize=(15, 6))
    for i in range(num_images):
        plt.subplot(2, num_images//2, i + 1)
        plt.imshow(images[i], cmap='binary')
        predicted = np.argmax(predictions[i])
        actual = labels[i]
        color = 'green' if predicted == actual else 'red'
        plt.title(f'Pred: {class_names[predicted]}\nActual: {class_names[actual]}', color=color)
        plt.axis('off')
    plt.tight_layout()
    plt.show()

# Class names for Fashion-MNIST
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

# Visualize predictions
visualize_predictions(model, test_batches, class_names, num_images=10)
```

---

## 📁 Recommended Repository Structure

```
tensorflow-deep-learning-guide/
│
├── README.md                    # This file
├── requirements.txt             # Dependencies
├── LICENSE                      # MIT License
│
├── notebooks/                   # Jupyter notebooks
│   ├── 01_tensors_and_operations.ipynb
│   ├── 02_data_pipelines.ipynb
│   ├── 03_building_networks.ipynb
│   ├── 04_training_and_validation.ipynb
│   ├── 05_overfitting_prevention.ipynb
│   ├── 06_saving_and_loading.ipynb
│   ├── 07_transfer_learning.ipynb
│   └── 08_gpu_acceleration.ipynb
│
├── scripts/                     # Python scripts
│   ├── train_mnist.py
│   ├── train_fashion_mnist.py
│   ├── evaluate_model.py
│   └── utils.py
│
├── models/                      # Saved models
│   ├── best_model.keras
│   └── checkpoints/
│
├── logs/                        # TensorBoard logs
│
└── data/                        # Data cache (auto-downloaded)
```

### requirements.txt

```
tensorflow==2.16.1
tensorflow-datasets==4.9.0
tensorflow-hub==0.15.0
numpy==1.24.3
matplotlib==3.7.1
jupyter==1.0.0
```

---

## 🚦 Quick Reference

### Common TensorFlow Operations

```python
# Create tensors
tf.constant([1, 2, 3])
tf.zeros((2, 3))
tf.ones((2, 3))
tf.random.normal((2, 2))
tf.random.uniform((2, 2))

# Tensor operations
tf.add(a, b)           # Element-wise addition
tf.multiply(a, b)      # Element-wise multiplication
tf.matmul(a, b)        # Matrix multiplication
tf.reduce_sum(a)       # Sum of all elements
tf.exp(a)              # Element-wise exponential
tf.reshape(a, (2, -1)) # Reshape tensor

# Activation functions
tf.nn.relu(x)
tf.nn.softmax(x)
tf.nn.sigmoid(x)
tf.nn.tanh(x)

# Loss functions
tf.keras.losses.MeanSquaredError()
tf.keras.losses.SparseCategoricalCrossentropy()
tf.keras.losses.BinaryCrossentropy()
```

---

## 📚 Resources

### Official Documentation
- [TensorFlow 2 Guide](https://www.tensorflow.org/guide)
- [Keras API Reference](https://keras.io/api/)
- [TensorFlow Datasets Catalog](https://www.tensorflow.org/datasets/catalog/)
- [TensorFlow Hub](https://tfhub.dev/)

### Tutorials
- [TensorFlow Core Tutorials](https://www.tensorflow.org/tutorials)
- [Keras Tutorials](https://keras.io/examples/)
- [Transfer Learning Guide](https://www.tensorflow.org/tutorials/images/transfer_learning)

### Community
- [TensorFlow GitHub](https://github.com/tensorflow/tensorflow)
- [Stack Overflow #tensorflow](https://stackoverflow.com/questions/tagged/tensorflow)
- [TensorFlow Blog](https://blog.tensorflow.org/)

---

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

```
MIT License

Copyright (c) 2024

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions...
```

---

## ⭐ Show Your Support

If this guide helped you learn TensorFlow, please give it a ⭐ on GitHub!

---

**Built with ❤️ and TensorFlow**
