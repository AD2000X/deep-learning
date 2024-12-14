# IMDB Sentiment Binary Classification: Transformer and BiLSTM
This project demonstrates techniques for building, training, and optimizing a neural network model to classify IMDB movie reviews as positive or negative.

## Table of Contents
1. [Natural Language Processing (NLP)](#1-natural-language-processing-nlp)
   - [Data Processing and Vectorization](#data-processing-and-vectorization)
   - [Feature Engineering](#feature-engineering)
2. [Deep Learning Techniques](#2-deep-learning-techniques)
   - [Model Architectures](#model-architectures)
   - [Regularization](#regularization)
   - [Optimization Methods](#optimization-methods)
3. [Model Evaluation and Hyperparameter Tuning](#3-model-evaluation-and-hyperparameter-tuning)
   - [Hyperparameter Tuning](#hyperparameter-tuning)
   - [Overfitting Detection](#overfitting-detection)
   - [Comparison Experiments](#comparison-experiments)
4. [Combining Traditional and Modern Methods](#4-combining-traditional-and-modern-methods)
5. [Resources and Tools](#5-resources-and-tools)
   - [Dataset](#dataset)
   - [Model Training Environment](#model-training-environment)
   - [Data Visualization](#data-visualization)

---

## 1. Natural Language Processing (NLP)

### Data Processing and Vectorization
1. **Text Vectorization (`TextVectorization`)**:
   - Utilized TensorFlow's `TextVectorization` layer to convert text into various numerical formats:
     - Bag-of-Words, multi-hot encoding, TF-IDF, and embedding vectors.
   - Supports the generation of N-grams (e.g., bigrams).
   - Processes text into formats suitable for deep learning models.

2. **Word Embedding**:
   - Used GloVe pre-trained word embeddings to initialize embedding layers, enabling transfer learning and helping models understand semantic information.

### Feature Engineering
1. **Bag-of-Words (BoW)**:
   - Represents text features using bigrams, capturing local contextual relationships between words.
2. **TF-IDF**:
   - Measures word importance within a document, combining term frequency and word rarity in the corpus to weight features.
3. **N-grams**:
   - Utilizes N-grams (especially bigrams) to capture sequential information in text, enhancing feature representation effectiveness.

---

## 2. Deep Learning Techniques

### Model Architectures
1. **Transformer Architecture**:
   - Implements a custom `TransformerEncoder` layer, supporting:
     - Multi-Head Attention for parallel processing of different parts of the text.
     - Positional embedding (`PositionalEmbedding`) to retain sequence information.
     - Feed-Forward Neural Networks for feature transformation.
   - Suitable for processing long text sequences and capturing global dependencies.

2. **Bidirectional LSTM (BiLSTM)**:
   - Processes sequential data from both forward and backward directions to capture long-term dependencies.
   - Requires fewer computational resources compared to Transformer models, making it suitable for resource-constrained environments.

3. **Dense Layers**:
   - Used for text classification tasks in Bag-of-Words and TF-IDF models.

4. **Pre-trained Word Embeddings (GloVe)**:
   - GloVe pre-trained embeddings contain semantic information learned from large-scale corpora, enhancing the model's generalization capabilities.

### Regularization
1. **Dropout**:
   - Randomly drops neurons during training to reduce overfitting.
2. **Early Stopping**:
   - Monitors validation loss and stops training early to prevent overfitting.
3. **Weight Initialization**:
   - Initialized model weights using pre-trained embeddings such as GloVe.

### Optimization Methods
1. **Gradient Optimizers**:
   - RMSprop and Adam optimizers were used, incorporating momentum and adaptive learning rate features.
2. **Learning Rate Scheduling**:
   - Custom learning rates and warm-up strategies were employed to improve convergence speed.
3. **Model Saving and Callbacks**:
   - `ModelCheckpoint` was used to save the best-performing model.

---

## 3. Model Evaluation and Hyperparameter Tuning

### Hyperparameter Tuning
1. Embedding dimensions.
2. Number of attention heads.
3. Batch size.
4. Number of LSTM units.
5. Transformer layers and dense layer units.

### Overfitting Detection
1. Monitored training and validation loss, as well as accuracy curves, to detect signs of overfitting.
2. Tested regularization methods (e.g., Dropout and Early Stopping) and data augmentation strategies.

### Comparison Experiments
1. Explored various model architectures (Transformer, LSTM, Bag-of-Words, TF-IDF) for IMDB movie review sentiment classification.
2. Compared test accuracy across different techniques to evaluate model performance.

---

## 4. Combining Traditional and Modern Methods
- **Traditional Methods**:
  - Bag-of-Words, TF-IDF.
- **Modern Methods**:
  - Transformer, LSTM, and pre-trained word embeddings (GloVe).

---

## 5. Resources and Tools

### Dataset
1. The IMDB movie review dataset was the primary data source.
2. Data preprocessing included removing irrelevant data (e.g., unlabeled reviews).

### Model Training Environment
1. Training was conducted on T4 GPUs, with a focus on the challenges of resource constraints and time efficiency.

### Data Visualization
1. Used Matplotlib and Seaborn to visualize data distribution and the training process (e.g., loss and accuracy curves).
2. Facilitated the analysis of overfitting and model performance.
