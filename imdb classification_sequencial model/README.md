# IMDB Sentiment Binary Classification: Sequencial Model

This project demonstrates techniques for building, training, and optimizing a neural network model to classify IMDB movie reviews as positive or negative.

## Table of Contents
1. [Data Processing Techniques](#data-processing-techniques)
2. [Deep Learning Architecture](#deep-learning-architecture)
3. [Model Optimization Techniques](#model-optimization-techniques)
4. [Regularization Techniques](#regularization-techniques)
5. [Performance Evaluation and Hyperparameter Tuning](#performance-evaluation-and-hyperparameter-tuning)
6. [Model Improvement Strategies](#model-improvement-strategies)
7. [Engineering Practices and Resource Management](#engineering-practices-and-resource-management)

---

## 1. Data Processing Techniques

### Dataset Loading and Splitting
- Used TensorFlow to load the IMDB dataset and split it into:
  - Training set: 15,000 samples
  - Validation set: 10,000 samples
  - Test set: 25,000 samples
- Filtered vocabulary based on word frequency, keeping only the top 10,000 most frequent words.

### Text Preprocessing
- Applied **multi-hot encoding** to convert text reviews into 10,000-dimensional binary feature vectors.

### Batch Processing
- Set an appropriate `batch_size` (e.g., 256) to balance memory usage and training efficiency.

---

## 2. Deep Learning Architecture

### Model Construction
- Built a multilayer fully connected neural network using TensorFlow/Keras's `Sequential` API.
- Defined:
  - Input layer: 10,000 dimensions
  - Hidden layer: 4 neurons
  - Output layer: 1 neuron

### Activation Functions
- Used `ReLU` for non-linearity.
- Tested `tanh` to observe its impact on performance.
- Used `sigmoid` in the output layer for binary classification.

### Loss Functions
- Tested:
  - **Binary cross-entropy** (`binary_crossentropy`)
  - **Mean squared error** (`mean_squared_error`)
- Analyzed the impact of these functions on learning outcomes.

---

## 3. Model Optimization Techniques

### Optimizer Selection
- Used:
  - **RMSProp**: Suitable for sparse data, adaptive learning rate.
  - **Adam**: Combines RMSProp and momentum advantages.

### Learning Rate Adjustment
- Experimented with learning rates:
  - `0.001`, `0.0005`, `0.0001`
- Identified optimal values for faster convergence.

### Momentum Adjustment
- Tested momentum parameters:
  - `0.9` and `0.5`
- Accelerated gradient descent using momentum.

---

## 4. Regularization Techniques

### L1 Regularization (Lasso)
- Penalized the absolute value of weights to control model complexity and prevent overfitting.
- Tested penalty strengths: `0.001` to `0.00001`.

### L2 Regularization (Ridge)
- Penalized the square of weights to suppress large weights and control complexity.
- Tested penalty strengths: `0.001` to `0.00001`.

### L1+L2 Regularization (Elastic Net)
- Combined L1 and L2 to balance feature selection and model generalization.

### Dropout
- Randomly “turned off” neurons during training to prevent reliance on specific features.
- Tested dropout rates: `0.01` to `0.5`.

### Early Stopping
- Dynamically stopped training based on validation performance (e.g., accuracy or loss) to prevent overfitting.

---

## 5. Performance Evaluation and Hyperparameter Tuning

### Grid Search
- Tested multiple hyperparameter combinations:
  - Learning rate
  - Regularization strength
  - Batch size
- Identified the best configuration.

### Performance Monitoring and Logging
- Monitored model performance using:
  - Accuracy
  - Loss metrics
- Logged training results in Pandas DataFrames for comparison and analysis.

### Visualization
- Used Matplotlib to plot learning curves, comparing training and validation:
  - Loss
  - Accuracy

---

## 6. Model Improvement Strategies

### Network Structure Adjustment
- Tested adding more hidden layers and increasing the number of neurons to improve model performance.
- Example: Increased to 4 hidden layers and included `tanh` activation functions.

### Automated Hyperparameter Tuning
- Introduced grid search to reduce manual tuning and increase experiment efficiency.

### Performance-Based Hyperparameter Refinement
- Adjusted hyperparameter configurations based on:
  - Learning curves
  - Validation performance (e.g., validation loss and accuracy).

---

## 7. Engineering Practices and Resource Management

### Memory Management
- Cleared memory using:
  - TensorFlow's `clear_session()`
  - Python's garbage collection (`gc`)
- Optimized resource usage.

### Modular Design
- Encapsulated functionalities into independent functions:
  - Training
  - Performance evaluation
  - Grid search

### Dynamic Training Control
- Used callback functions (e.g., `EarlyStopping`) to adaptively adjust training processes.
