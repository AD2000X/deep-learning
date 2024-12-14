# Comparison of CW1 and CW2: Similarities and Differences

## Table of Contents
1. [Similarities](#similarities)
   - [Data Processing Techniques](#data-processing-techniques)
   - [Deep Learning Techniques](#deep-learning-techniques)
   - [Model Evaluation and Hyperparameter Tuning](#model-evaluation-and-hyperparameter-tuning)
   - [Engineering Practices](#engineering-practices)
2. [Differences](#differences)
   - [Data Processing Techniques](#data-processing-techniques-1)
   - [Feature Engineering](#feature-engineering)
   - [Deep Learning Techniques](#deep-learning-techniques-1)
   - [Regularization Techniques](#regularization-techniques)
   - [Optimization Methods](#optimization-methods)
   - [Model Evaluation and Tuning](#model-evaluation-and-tuning)
   - [Resources and Tools](#resources-and-tools)
3. [Summary Table](#summary-table)
4. [Conclusion](#conclusion)

---

## 1. Similarities

### Data Processing Techniques
- **Dataset Splitting**:
  - Both CW1 and CW2 use the IMDB dataset, splitting it into training, validation, and test sets.
- **Text Representation**:
  - Both use multi-hot encoding for representing text as binary feature vectors.
  - CW2 extends this by incorporating more advanced vectorization techniques like Bag-of-Words (BoW), TF-IDF, and word embeddings.

### Deep Learning Techniques
- **Model Architectures**:
  - Both CW1 and CW2 use dense layers for text classification tasks.
  - CW2 expands on this by introducing advanced architectures such as Transformer and Bidirectional LSTM (BiLSTM).
- **Activation Functions**:
  - Both tasks use ReLU for hidden layers and sigmoid for binary classification.
- **Regularization**:
  - Both CW1 and CW2 apply dropout and early stopping to prevent overfitting.
- **Optimization Methods**:
  - Both utilize RMSprop and Adam optimizers with learning rate adjustments.

### Model Evaluation and Hyperparameter Tuning
- **Grid Search**:
  - Both CW1 and CW2 employ grid search for hyperparameter tuning (e.g., learning rates, regularization strength, batch size).
- **Overfitting Detection**:
  - Both monitor training and validation loss curves to identify overfitting.
- **Visualization**:
  - Both use Matplotlib to visualize training and validation performance metrics.

### Engineering Practices
- **Resource Optimization**:
  - CW1 and CW2 use callback functions (e.g., early stopping) and memory management techniques to optimize GPU resources.

---

## 2. Differences

### Data Processing Techniques
- **Vocabulary Handling**:
  - CW1 filters vocabulary by word frequency, retaining the top 10,000 words.
  - CW2 uses the TextVectorization layer, supporting more advanced formats like N-grams and embedding vectors.
- **Word Embeddings**:
  - CW1 does not use pre-trained embeddings.
  - CW2 incorporates GloVe pre-trained embeddings for semantic understanding and transfer learning.

### Feature Engineering
- **CW1**:
  - Primarily uses multi-hot encoding to create binary feature vectors.
- **CW2**:
  - Expands feature engineering to include:
    - Bag-of-Words (BoW) for local context.
    - TF-IDF to weight term importance.
    - N-grams for capturing sequential information.

### Deep Learning Techniques
- **CW1**:
  - Uses a simple feed-forward neural network with one hidden layer.
- **CW2**:
  - Implements more sophisticated architectures:
    - TransformerEncoder for long text sequences and global dependencies.
    - Bidirectional LSTM for capturing long-term dependencies.
    - Positional embeddings for retaining sequence information.

### Regularization Techniques
- **CW1**:
  - Applies L1, L2, and Elastic Net regularizations for weight penalties.
- **CW2**:
  - Uses weight initialization with pre-trained embeddings like GloVe.

### Optimization Methods
- **Learning Rate Strategies**:
  - CW1 experiments with static learning rates.
  - CW2 incorporates advanced learning rate scheduling and warm-up strategies.
- **Model Saving**:
  - CW2 uses `ModelCheckpoint` for saving the best-performing models, which is absent in CW1.

### Model Evaluation and Tuning
- **Hyperparameters**:
  - CW1 focuses on simpler hyperparameters such as learning rate, momentum, and regularization strength.
  - CW2 introduces additional parameters like embedding dimensions, attention heads, and LSTM units.
- **Comparison Experiments**:
  - CW1 primarily tests variations of the feed-forward neural network.
  - CW2 compares multiple architectures (Transformer, BiLSTM, BoW, TF-IDF).

### Resources and Tools
- **Visualization Tools**:
  - CW1 uses Matplotlib for basic performance visualization.
  - CW2 uses both Matplotlib and Seaborn for advanced data distribution and performance analysis.
- **Model Training Environment**:
  - CW2 explicitly leverages T4 GPUs and addresses resource constraints for larger architectures like Transformers.

---

## 3. Summary Table

| **Category**               | **Similarities**                                                    | **Differences**                                                                                   |
|-----------------------------|--------------------------------------------------------------------|---------------------------------------------------------------------------------------------------|
| **Data Processing**         | Multi-hot encoding for binary vectors.                           | CW2 uses TextVectorization with BoW, TF-IDF, and GloVe embeddings.                               |
| **Feature Engineering**     | Both focus on text vectorization.                                | CW1 uses simple multi-hot encoding, while CW2 integrates N-grams, BoW, and TF-IDF.              |
| **Deep Learning Techniques**| Dense layers, ReLU activation, sigmoid for binary classification.| CW2 adds Transformer, BiLSTM, and positional embeddings.                                         |
| **Regularization**          | Dropout and early stopping for overfitting prevention.           | CW1 adds L1, L2, and Elastic Net penalties, while CW2 uses GloVe for weight initialization.      |
| **Optimization**            | RMSprop and Adam optimizers.                                     | CW2 incorporates learning rate schedules and `ModelCheckpoint` for saving models.               |
| **Evaluation and Tuning**   | Grid search and loss visualization.                              | CW2 introduces additional hyperparameters and compares multiple advanced architectures.          |
| **Resources and Tools**     | Matplotlib for visualizations.                                   | CW2 uses Seaborn for enhanced analysis and T4 GPUs for training large-scale models.              |

---

## 4. Conclusion
- **CW1** is focused on basic feed-forward architectures and simpler regularization/optimization techniques, suitable for small-scale experiments and prototyping.
- **CW2** integrates modern NLP techniques and sophisticated deep learning architectures, making it more suitable for larger datasets and advanced tasks like contextual understanding and long-term dependencies.
