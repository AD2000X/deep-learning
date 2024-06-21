# %% [markdown]
# Training and Fine-tuning Workflow:
# 
# 1. Define the task  
# 
# 2. Validation recipe  
# 
# 3. Measure of success  
# 
# 4. Training structure:
# 
# >4.1. Prepare dataset  
# 
# >4.2. Evaluation:  
#     >>4.2.1 Divide dataset  
#     4.2.2 Data representation  
#     4.2.3 The time arrow of time  
#     4.2.4 Redundancy in data  
# 
# 5. Beat a baseline.  
# 
# 6. Develop an overfitting model:  
# 
# 
# > 6.1. Model without overfitting:  
#     >>6.1.1. Scale up: Develop a model that overfits  
#       >>>a. Add layers  
#       b. Make the layers bigger  
#       c. More epochs  
# 
# >6.2. Model with overfitting:  
#     >>6.2.1. Improving model fit (fine tuning strategy):   
#     >>>a. Optimize learning rate  
#     b. Increasing model capacity:  
#     >>>>i. Add layers  
#     ii. Increase batch size    
#     iii. Train for more epochs  
# 
# 7. Improving generalization:
# 
# >7.1 Dataset curation  
# 
# >7.2 Feature engineering  
# 
# >7.3 Using early stopping  
# 
# >7.4 Regularizing your model  
# 
# >>7.4.1 Reducing the network's size  
# >>7.4.2 Adding weight regularization  
# >>7.4.3 Adding Dropout  
# 
# 8. Advanced tuning:  
# 
# >8-1 Use mse as Loss Function instead  
# 
# >8-2 Use tanh as Activate Function instead  
# 
# >8-3 Add dense layer  
# 
# >8-4 Increase or decrease neurons  
# 
# 9. Final test model  
# 
# 10. Conclusion  
# 
# >Training Model Evaluation Criteria:  
# Generalization  
# Overfitting  
# Model Fit  
# 
# 11. Further test
# 
# 12. Postscript  
# 
# 13. Referencs

# %% [markdown]
# 
# 
# ---------------
# 
# 
# 
# 

# %% [markdown]
# # 1. Define the task
# The untimate goal of machine learning is to generalize the model and apply it to unseen data so that it can predict the future  as well as possible. The target of this research is training a model that can classifying movie reviews, positive or negative, by binary classification.

# %% [markdown]
# # 2. Dataset chosen(validation recipe)
# The IMDB (Internet Movie Database) dataset is a public dataset widely used in machine learning and natural language processing, especially in sentiment analysis tasks.
# 
# It is mainly used for binary sentiment classification tasks, determining whether a movie review, which are written by real users, is positive or negative. The dataset contains 50,000 reviews, split into a training set and a test set of 25,000 reviews each and each review is tagged as "positive" or "negative." The number of both sides reviews is approximately equal.
# 
# From Keras.io: "Words are indexed by overall frequency in the dataset, so that for instance the integer "3" encodes the 3rd most frequent word in the data. This allows for quick filtering operations such as: "only consider the top 10,000 most common words, but eliminate the top 20 most common words"."

# %%
# Codes are original from Jérémie's lecture and generated ChatGPT (ver. Plugins).
# Import statements
# Dataset
from tensorflow.keras.datasets import imdb

# System
import os
import gc
import sys
import time
import traceback

# Data processing
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import pandas as pd

# Regularizations
from tensorflow.keras import models, layers, losses, optimizers, regularizers
from tensorflow.keras.callbacks import EarlyStopping

# %% [markdown]
# #### A negative review  
# ##### # Codes originally from CW 1 example.

# %%
# Codes are original from Jérémie's lecture and generated ChatGPT (ver. Plugins).

# Load the dataset
((train_data, train_labels), (test_data, test_labels)) = tf.keras.datasets.imdb.load_data(num_words=10000)

# Print the shapes
print("Train data shape:", train_data.shape)
print("Test data shape:", test_data.shape)


# %%
print(train_data[100]) # how a text looks like: a list of numbers!

# %% [markdown]
# #### A negative review  
# ##### # Codes originally from CW 1 example.

# %%
word_index = tf.keras.datasets.imdb.get_word_index()                   # get words → index
reverse_word_index = {value:key for key,value in word_index.items()}   # reverse: index → word (Python dict comprehension)
decoded_review = ' '.join(
    [                                      # get() works like [] but you can set a default
        reverse_word_index.get(i - 3, '?') # value if the key isn't found -3 because the first
        for i in train_data[100]           # three slots in the vocab are 0: "padding",
    ]                                      #                              1: "start of sequence",
)                                          #                              2: "unknown"

# %%
print(decoded_review)

# %%
print(train_labels[100]) # 0: negative

# %% [markdown]
# #### A positive review
# ##### # Codes come from CW 1 example.
# 
# The question marks are in place of words outside the 10000 word dictionary.

# %%
# let's try another one
decoded_review = ' '.join([reverse_word_index.get(i - 3, '?') for i in train_data[83]])
print(decoded_review, '\n\nlabel: ', train_labels[83])

# %%
# Codes below are generated by ChatGPT Plugins

# download data: top 10000 most frequent words only, discard rare words
(train_data, train_labels), (test_data, test_labels) = tf.keras.datasets.imdb.load_data(num_words=10000)

# Print the shape of training and test data
print(train_data.shape, test_data.shape)

# Combine training and test labels to count the entire data set
all_labels = np.concatenate((train_labels, test_labels))

# Count the number of positive and negative comments
positive_reviews = np.sum(all_labels == 1)
negative_reviews = np.sum(all_labels == 0)

# Print the number of positive and negative comments
print(f"Number of positive reviews: {positive_reviews}")
print(f"Number of negative reviews: {negative_reviews}")

# %% [markdown]
# # 3. Measure of success, 4. Training structures.

# %% [markdown]
# At the first, we are going to divide our dataset into training set, validation set, and test sest. the first evaluation metric is our model can achieve over 50% accuracy on the validation set. After that, we aim to develop an overfitting model. Once our the model are able to be overfitting, the next step is improving model fit, including adjust learning rate, introduce weight regularization, and increasing model capacity.

# %% [markdown]
# ### Training evaluation

# %% [markdown]
# The following code is to reduce the uncertainty caused by random number generation when using TensorFlow and NumPy so that the results of tuning or model training can be better reproducibility. This means that only error information will be printed and warning and information logs will not be displayed. By doing so, we can reduce console information redundancy, focusing more on training and critical error information.

# %%
# Codes are from Jérémie's course "04.1.classifying-movie-reviews-imdb".

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
tf.random.set_seed(23)
np.random.seed(23)

# %% [markdown]
# ## Data Preprocess

# %% [markdown]
# ### Multi-hot encoding  
# Convert a sequence of integers to one-hot encoding so that we can convert words or tokens into a numerical form(binary matrices) that can be processed by a neural network. Suitable for processing large number of categories but each instance belongs to only one category.

# %%
# Codes below are from Dr.Jérémie's course "04.1.classifying-movie-reviews-imdb".

# Load IMDB dataset
(train_data, train_labels), (test_data, test_labels) = tf.keras.datasets.imdb.load_data(num_words=10000)

def vectorize_sequences(sequences, dimension=10000):
    results = np.zeros((len(sequences), dimension))
    for i, sequence in enumerate(sequences):
        results[i, sequence] = 1.0
    return results

# Preprocess dataset
# Vectorization process on the training data and test data, ensuring consistency in the format and processing.
x_train = vectorize_sequences(train_data)
x_test = vectorize_sequences(test_data)
y_train = np.asarray(train_labels).astype("float32")
y_test = np.asarray(test_labels).astype("float32")

# Simple holdout validation: Spilit dataset into train set and validation set
partial_x_train = x_train[10000:]
partial_y_train = y_train[10000:]
x_val = x_train[:10000]
y_val = y_train[:10000]

# %% [markdown]
# ### Split dataset into training, validation, and test set.

# %%
# Codes below are from W.Jérémie's course "04.1.classifying-movie-reviews-imdb".
# Calculate proportions of parts(by ChatGPT Plugins)
total_samples = len(x_train) + len(x_test)
train_ratio = len(partial_x_train) / total_samples
val_ratio = len(x_val) / total_samples
test_ratio = len(x_test) / total_samples

# Codes below are generated by ChatGPT Plugins
print(f"Training set ratio: {train_ratio:.2f}")
print(f"Validation set ratio: {val_ratio:.2f}")
print(f"Test set ratio: {test_ratio:.2f}")

# %% [markdown]
# ### Training History

# %%
# Codes are original from Jérémie's "lab-3-IMDB". Modified by ChatGPT(ver. Plugins).
# Use for training the model multiple times, comparing different training runs, and tuning hyperparameters.

def train_and_save_history(model, x_train, y_train, x_val, y_val, epochs, batch_size, all_histories, callbacks=None):
    history = model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size, validation_data=(x_val, y_val), callbacks=callbacks)
    all_histories.append(history.history)
    return history.history

# %% [markdown]
# ### Plot History

# %%
# Codes are original from Jérémie's "lab-3-IMDB". Modified by ChatGPT(ver. Plugins).
# Prompts included "Help me optimise the chart" and "Change the dot color", but I have to provide the "Training History" function code, "Model" function code, and "Metric Dataframe" function code, or the variable will be randomly named by ChatGPT (it's a disaster).

def plot_history(history):
    acc = history['accuracy']
    val_acc = history['val_accuracy']
    loss = history['loss']
    val_loss = history['val_loss']
    epochs = range(1, len(acc) + 1)

    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(epochs, loss, 'ro-', label='Training loss')  # 训练损失的红色点和实线
    plt.plot(epochs, val_loss, 'bo-', label='Validation loss')  # 验证损失的蓝色点和实线
    plt.title('Training and validation loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(epochs, acc, 'ro-', label='Training acc')  # 训练准确率的红色点和实线
    plt.plot(epochs, val_acc, 'bo-', label='Validation acc')  # 验证准确率的蓝色点和实线
    plt.title('Training and validation accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.tight_layout()
    plt.show()


# %% [markdown]
# ## Metric Dataframe

# %%
# Idea originally from CW 1 example. Modified by ChatGPT(ver. Plugins).
# Codes are generated together with the "Main function" function, "Model" function, and "Training History" function by ChatGPT (ver. Plugins).

def generate_performance_dataframe(model: tf.keras.Model, history: dict) -> pd.DataFrame:
    
    model_name = model.name if model.name else 'Unnamed Model'
    num_layers = len(model.layers)

    # first dense units
    units = model.layers[0].units

    # Initialize regularization and dropout rate
    l1_reg = None
    l2_reg = None
    dropout_rate = None

    # Traverse each layer to extract L1, L2 regularization parameters and dropout rate
    for layer in model.layers:
        if hasattr(layer, 'rate'):
            dropout_rate = layer.rate
        if hasattr(layer, 'kernel_regularizer') and layer.kernel_regularizer is not None:
            regularizer = layer.kernel_regularizer.get_config()
            l1_reg = regularizer.get('l1')
            l2_reg = regularizer.get('l2')

    # Optimizer configuration
    optimizer_config = model.optimizer.get_config()
    learning_rate = optimizer_config.get('learning_rate')
    momentum = optimizer_config.get('momentum', None)

    # Extract minimum verification loss and maximum verification accuracy from historical dictionary
    min_val_loss_value = np.min(history['val_loss'])
    max_val_acc_value = np.max(history['val_accuracy'])

    # Create a DataFrame containing actual model performance metrics
    data = {
        'Model': [model_name],
        'Layers': [num_layers],
        'Units': [units],
        'LR': [learning_rate],
        'Momentum': [momentum],
        'L1': [l1_reg],
        'L2': [l2_reg],
        'Dropout': [dropout_rate],
        'Min.V.Loss': [min_val_loss_value],
        'Max.V.Acc': [max_val_acc_value]
    }

    df = pd.DataFrame(data)

    return df


# %% [markdown]
# ## Clean memory

# %%
# Codes are original from Jérémie's lecture "04.1.classifying-movie-reviews-imdb".
# Memory clean-up for Jupyter
# Slightly modified from fast.ai utils: https://github.com/fastai/course22p2/blob/df9323235bc395b5c2f58a3d08b83761947b9b93/miniai/init.py#L31

def clean_ipython_hist():
    # Code in this function mainly copied from IPython source
    if not 'get_ipython' in globals(): return
    ip = get_ipython()
    user_ns = ip.user_ns
    ip.displayhook.flush()
    pc = ip.displayhook.prompt_count + 1
    for n in range(1, pc): user_ns.pop('_i'+repr(n),None)
    user_ns.update(dict(_i='',_ii='',_iii=''))
    hm = ip.history_manager
    hm.input_hist_parsed[:] = [''] * pc
    hm.input_hist_raw[:] = [''] * pc
    hm._i = hm._ii = hm._iii = hm._i00 =  ''

def clean_tb():
    # h/t Piotr Czapla
    if hasattr(sys, 'last_traceback'):
        traceback.clear_frames(sys.last_traceback)
        delattr(sys, 'last_traceback')
    if hasattr(sys, 'last_type'): delattr(sys, 'last_type')
    if hasattr(sys, 'last_value'): delattr(sys, 'last_value')

def clean_mem():
    clean_tb()
    clean_ipython_hist()
    gc.collect()
    tf.keras.backend.clear_session()
    time.sleep(2)

# %% [markdown]
# ## Main function

# %%
# Codes are generated together with the "Metrics DataFrame" function, "Model" function, and "Training History" function by ChatGPT (ver. Plugins).
# Keep track of the training process and results of various models.

# main function
if __name__ == "__main__":
    all_histories = [] # save all training history data in lists
    models = {} # save all models and history data

# %% [markdown]
# ## According to the principle of Occam’s razor: Given two explanations for something, the explanation most likely to be correct is the simplest one—the one that makes fewer assumptions. We start with the simplest model.

# %% [markdown]
# # 5. Beat the baseline, and 6. Develop an overfitting model:  
# 
# ## # All models' codes are original from Jérémie's lectures and lab. Modified by ChatGPT (ver. Plugins), during 5/11-12/11.

# %% [markdown]
# # Model_01:
# lyers=2  
# epochs=20  
# batch_size=256

# %%
# Define model
model = tf.keras.models.Sequential([
tf.keras.layers.Dense(4, activation='relu', input_shape=(10000,)),
tf.keras.layers.Dense(1, activation='sigmoid')
])

# Complie model
model.compile(optimizer=optimizers.RMSprop(learning_rate=0.001),
              loss=losses.binary_crossentropy,
              metrics=['accuracy'])

# Train model and store history
history = train_and_save_history(model, partial_x_train, partial_y_train, x_val, y_val, 20, 256, all_histories)

# Evaluate model performance
results = model.evaluate(x_test, y_test)
print(f"Test Loss: {results[0]}, Test Accuracy: {results[1]}")

# Plot both training and validation loss and accuracy of the last training
plot_history(all_histories[-1])

# %% [markdown]
# # Performance_Model_01:

# %%
df = generate_performance_dataframe(model, history)
print(df)

clean_mem()

# %% [markdown]
# # Model_01 Summary:  
# From the above plot, we can observe that the training set accuracy value and loss value are getting better during the training period. However, the validation set accuracy reach the highest at epoch 6(0.8906) and minimum loss 0.272166 at epoch 5, but following by the increasing loss and oscillate back and forth on the accuracy. Due to the validation accuracy value outrun the baseline(50%), we can also infer that the representational power is enough. The model gets better on the training data, but stops improving on the unseen validation data, representing that it is a little bit overfitting.  
# 
# # Following adjustment strategies:  
# Since our model is capable of overfitting, we decide to perform further adjustment.  
# Aim: Find a model that beats the validation accuracy value 0.8910, or beat overall training performance(Test Loss: 0.4139, Test Accuracy: 0.8650).  
# Method: We modify only one hyperparameter at one time till we find a satisified model.  
# Steps:  
# 1. add 1 more layer  
# 2. higher and lower Learning rate

# %% [markdown]
# # Model_02
# layers=3 (add 1 more here)  
# epochs=20  
# batch_size=256

# %%
# Define model
model = tf.keras.models.Sequential([
tf.keras.layers.Dense(4, activation='relu', input_shape=(10000,)),
tf.keras.layers.Dense(1, activation='relu'),
tf.keras.layers.Dense(1, activation='sigmoid')
])

# Complie model
model.compile(optimizer=optimizers.RMSprop(learning_rate=0.001),
              loss=losses.binary_crossentropy,
              metrics=['accuracy'])

# Train model and store history
history = train_and_save_history(model, partial_x_train, partial_y_train, x_val, y_val, 20, 256, all_histories)

# Evaluate model performance
results = model.evaluate(x_test, y_test)
print(f"Test Loss: {results[0]}, Test Accuracy: {results[1]}")

# Plot both training and validation loss and accuracy of the last training
plot_history(all_histories[-1])

# %% [markdown]
# # Performance_Model_02

# %%
df = generate_performance_dataframe(model, history)
print(df)

clean_mem()

# %% [markdown]
# # Model_02 Summary  
# The model's accuracy gradually improves during training, but starts to fluctuate on validation loss value, but remains stable on validation accuracy value, meaning it is overfitting and perform less well on unseen data. Moreover, the highest validation accuracy value of Model_02 is 0.833, lower than that of Model_01, which is 0.8912. According to our strategy, we are going to modify other hyperparameter and leave Model_02 alone as for now.

# %% [markdown]
# # Model_03
# layer=2  
# epochs=20  
# batch_size=256  
# learning rate=0.0001(new)

# %%
# Define model
model = tf.keras.models.Sequential([
tf.keras.layers.Dense(4, activation='relu', input_shape=(10000,)),
tf.keras.layers.Dense(1, activation='sigmoid')
])

# Complie model
model.compile(optimizer=optimizers.RMSprop(learning_rate=0.0001),
              loss=losses.binary_crossentropy,
              metrics=['accuracy'])

# Train model and store history
history = train_and_save_history(model, partial_x_train, partial_y_train, x_val, y_val, 20, 256, all_histories)

# Evaluate model performance
results = model.evaluate(x_test, y_test)
print(f"Test Loss: {results[0]}, Test Accuracy: {results[1]}")

# Plot both training and validation loss and accuracy of the last training
plot_history(all_histories[-1])

# %% [markdown]
# # Performance_Model_03:  

# %%
df = generate_performance_dataframe(model, history)
print(df)

clean_mem()

# %% [markdown]
# # Model_03 Summary:   
# The result of the tuning of the learning rate proves that the key to gradient descent parameters is learning rate(5-3-1, DLWP) is true.
# Comparing the overall performance between Model_01 and Model_03, the Model_03 overall test Loss is 0.3853, and the overall test accuracy is 0.8690, which is all better than the Model_01 test loss of 0.4392, test accuracy is 0.8650. The smooth learning curve of training loss and accuracy and that of validation loss and accuracy is on a satisfactory level. Model_03 test loss decreased from 0.6788 to 0.3433, showing that the model gradually improves its predictions. Model_03 validation loss dropped from 0.6648 to 0.3752, showing that the model performs well on unseen data without significant overfitting. Model_03 training accuracy increased from 0.5940 to 0.9020, indicating that model's performance on training set is steady improving. The validation accuracy increased from 0.6488 to 0.8746, demonstrating the well generalization ability of the model on unseen data.
# 
# Moreover, considering of overfitting, both loss curves were steadily decreased. There were no signs of overfitting such as validation loss decreased at first but later on increased or traning accuracy is siginificantly higher than validation accuracy.
# 
# Furthermore, the steady increase in accuracy and steady decrease in loss provide evidence that the model is steady learning and is neither oscillated nor stagnated by the learning rate.  
# 
# Overall, even though the highest validation accuracy rate 0.8746 is slightly lower than the Model_01 validation accuracy value of 0.8910, we still can choose Model_03 as the fundamental for the further training process. But before we move on, we want to check the affection of learning rate.  

# %% [markdown]
# # Apparently, tuning the learning rate has a significant impact on model learning, we can further tuning on different learning rates, but before we start to verify the learning rate, it is worth noting that, all the loss values are decreasing and all the accuracy values are increasing, so we can increase the epoch to 30 for a test first.

# %% [markdown]
# # Model_03-1(increase epoch)    
# 
# layer=2  
# epochs=30(new, from 20 to 30)  
# batch_size=256  
# learning rate=0.0001  
# 

# %%
# Define model
model = tf.keras.models.Sequential([
tf.keras.layers.Dense(4, activation='relu', input_shape=(10000,)),
tf.keras.layers.Dense(1, activation='sigmoid')
])

# Complie model
model.compile(optimizer=optimizers.RMSprop(learning_rate=0.0001),
              loss=losses.binary_crossentropy,
              metrics=['accuracy'])

# Train model and store history
history = train_and_save_history(model, partial_x_train, partial_y_train, x_val, y_val, 30, 256, all_histories)

# Evaluate model performance
results = model.evaluate(x_test, y_test)
print(f"Test Loss: {results[0]}, Test Accuracy: {results[1]}")

# Plot both training and validation loss and accuracy of the last training
plot_history(all_histories[-1])

# %% [markdown]
# #Model_03-1 Summary:  
# We can easily find that all values ​​are stably getting better, but we also can see that the amplitude has become smaller. We increase epoch to 50 and see if it will overfitting.

# %% [markdown]
# 

# %%
df = generate_performance_dataframe(model, history)
print(df)

clean_mem()

# %% [markdown]
# # Model_03-2(increase epoch)  
# 
# layer=2  
# epochs=50(new, from 30 to 50)  
# batch_size=256  
# learning rate=0.0001  
# 

# %%
# Define model
model = tf.keras.models.Sequential([
tf.keras.layers.Dense(4, activation='relu', input_shape=(10000,)),
tf.keras.layers.Dense(1, activation='sigmoid')
])

# Complie model
model.compile(optimizer=optimizers.RMSprop(learning_rate=0.0001),
              loss=losses.binary_crossentropy,
              metrics=['accuracy'])

# Train model and store history
history = train_and_save_history(model, partial_x_train, partial_y_train, x_val, y_val, 50, 256, all_histories)

# Evaluate model performance
results = model.evaluate(x_test, y_test)
print(f"Test Loss: {results[0]}, Test Accuracy: {results[1]}")

# Plot both training and validation loss and accuracy of the last training
plot_history(all_histories[-1])

# %% [markdown]
# # Model_03-2 Summary:  
# The model has gone through 50 training stages (epochs), representing a complete traversal of the model over the entire training dataset.
# 
# However, it starts to oscillate on epoch 42 and seems reaching its learning limitation among epochs 48, 49, and 50. Other possible explanations include the need to be specific fine-tuning or the dataset or model limitations.
# 
# Overall, we decided to stop here because the distance of training and validation loss curve is enlarged, it is a sign of overfitting. Therefore we shall just leave Model_03-2 here for now.

# %%
df = generate_performance_dataframe(model, history)
print(df)

clean_mem()

# %% [markdown]
# # Learning rate tuning summary:  
# Model_04(learning rate=0.0005):  
# Compared with Model_03, Model_04 has better accuracy and loss during training but is accompanied by a large gap between test accuracy, 0.9727, and validation accuracy, 0.8921, indicating the sign of overfitting. Moreover, the validation accuracy rate reached the highest value, 0.8905 at epoch 10, then roughly remained at 0.88, illustrating the sign of overfitting as well. Last, it performed well during the training process, and its accuracy gradually improved, showing the capability on certain levels of generalizations.
# 
# Model_05(learning rate=0.0004):  
# It appears the same problem(overfitting, but earlier, on epoch 5) as Model_04, but the overall performance is slightly better than Model_04.  
# 
# Model_06(learning rate=0.0003):  
# Also, it started overfitting on epoch 6) as Model_04, but the overall performance is slightly worse than Model_04.  
# 
# Model_07(learning rate=0.0002):  
# Apparently, the learning rate tuning shall stop here because the overall performance of test loss and validation were separately increased by 17% and dropped by 4%.  
# 
# ### Hence, Model_04 will be choosed for further training.

# %% [markdown]
# # Model_04(Learning rate tuning)  
# layer=2  
# epochs=20  
# batch_size=256  
# learning rate=0.0005(new)

# %%
# Define model
model = tf.keras.models.Sequential([
tf.keras.layers.Dense(4, activation='relu', input_shape=(10000,)),
tf.keras.layers.Dense(1, activation='sigmoid')
])

# Complie model
model.compile(optimizer=optimizers.RMSprop(learning_rate=0.0005),
              loss=losses.binary_crossentropy,
              metrics=['accuracy'])

# Train model and store history
history = train_and_save_history(model, partial_x_train, partial_y_train, x_val, y_val, 20, 256, all_histories)

# Evaluate model performance
results = model.evaluate(x_test, y_test)
print(f"Test Loss: {results[0]}, Test Accuracy: {results[1]}")

# %%
# Plot both training and validation loss and accuracy of the last training
plot_history(all_histories[-1])

# %% [markdown]
# # Performance_Model_04:

# %%
df = generate_performance_dataframe(model, history)
print(df)

clean_mem()

# %% [markdown]
# # Model_05(Learning rate tuning)    
# layer=2  
# epochs=20  
# batch_size=256  
# learning rate=0.0004(new)

# %%
# Define model
model = tf.keras.models.Sequential([
tf.keras.layers.Dense(4, activation='relu', input_shape=(10000,)),
tf.keras.layers.Dense(1, activation='sigmoid')
])

# Complie model
model.compile(optimizer=optimizers.RMSprop(learning_rate=0.0004),
              loss=losses.binary_crossentropy,
              metrics=['accuracy'])

# Train model and store history
history = train_and_save_history(model, partial_x_train, partial_y_train, x_val, y_val, 20, 256, all_histories)


# Evaluate model performance
results = model.evaluate(x_test, y_test)
print(f"Test Loss: {results[0]}, Test Accuracy: {results[1]}")

# Plot both training and validation loss and accuracy of the last training
plot_history(all_histories[-1])

# %% [markdown]
# # Performance_Model_05:

# %%
df = generate_performance_dataframe(model, history)
print(df)

clean_mem()

# %% [markdown]
# # Model_06(Learning rate tuning)    
# layer=2  
# epochs=20  
# batch_size=256  
# learning rate=0.0003(new)

# %%
# Define model
model = tf.keras.models.Sequential([
tf.keras.layers.Dense(4, activation='relu', input_shape=(10000,)),
tf.keras.layers.Dense(1, activation='sigmoid')
])

# Complie model
model.compile(optimizer=optimizers.RMSprop(learning_rate=0.0003),
              loss=losses.binary_crossentropy,
              metrics=['accuracy'])

# Train model and store history
history = train_and_save_history(model, partial_x_train, partial_y_train, x_val, y_val, 20, 256, all_histories)

# Evaluate model performance
results = model.evaluate(x_test, y_test)
print(f"Test Loss: {results[0]}, Test Accuracy: {results[1]}")

# Plot both training and validation loss and accuracy of the last training
plot_history(all_histories[-1])

# %% [markdown]
# # Performance_Model_06:

# %%
df = generate_performance_dataframe(model, history)
print(df)

clean_mem()

# %% [markdown]
# # Model_07(Learning rate tuning)    
# layer=2  
# epochs=20  
# batch_size=256  
# learning rate=0.0002(new)

# %%
# Define model
model = tf.keras.models.Sequential([
tf.keras.layers.Dense(4, activation='relu', input_shape=(10000,)),
tf.keras.layers.Dense(1, activation='sigmoid')
])

# Complie model
model.compile(optimizer=optimizers.RMSprop(learning_rate=0.0002),
              loss=losses.binary_crossentropy,
              metrics=['accuracy'])

# Train model and store history
history = train_and_save_history(model, partial_x_train, partial_y_train, x_val, y_val, 20, 256, all_histories)

# Evaluate model performance
results = model.evaluate(x_test, y_test)
print(f"Test Loss: {results[0]}, Test Accuracy: {results[1]}")

# Plot both training and validation loss and accuracy of the last training
plot_history(all_histories[-1])

# %% [markdown]
# # Performance_Model_07:

# %%
df = generate_performance_dataframe(model, history)
print(df)

clean_mem()

# %% [markdown]
# # Next step: Improved the generalization.  
# 
# Because Model_04 demonstrated the early sign of overfitting but the highest validation accuracy rate(0.8905), we decided to introduce L1, L2, and L1+L2 regularization.
# 
# Few strategies we can choose:  
# 1. Dataset curation:  
#     Including Data Collection, Data Cleansing, Data annotation, Data augmentation, Data splitting, and Data maintenance and updates. We already choose data splitting at the very beginning, and the rest methods have to be taken. Let's take a look for the next one.
# 
# 2. Feature engineering:  
#     This is out of scope for now(out of my abilitie for now.). Next one.
# 
# 3. Using early stopping:  
#     Early stopping is usually adopted in the model that shows the evidences of overfitting. We can save this option for tuning Model_04. Considering of far exceed the minimum number of parameters required to fit the underlying manifold of the data, over-parameterization of deep learning models is common. Furthermore, the incompletely fitting model is usual as well, because a perfect fit will not generalize well. Therefore, the early stoppiing can be use on before the reaching the minimum possible training loss to avoid overfitting, meaning interrupt training at some point. We can conclude that the most effective way of finding the most generalizable fitting point during training, which means locating the exact boundary between the underfitting and overfitting curves. Since Model_03 has no overfitting issue, we can test the early stopping on Model_04.
#     
# 4. Regularizing your model:  
# 
#     4.1 Adjusting the network's size:  
#     Our network currently contains only 2 layers and 4 units, so this can be an option for Model_04.
# 
#     4.2 Adding weight regularization: L1, L2, L1+L2:  
#     L1 regularization (Lasso regularization) adds the sum of the absolute values ​​of the weights to the loss function. This usually applies to when the number of features is large and only a few are important. It also can reduce unnecessary features When model interpretability is necessary.  
# 
#     L2 regularization (Ridge regularization) adds the sum of squared weights to the loss function. This usually applies when data features are highly correlated and all features are required to determine the output.  
# 
#     The L1 + L2(Elastic Net) combines the characteristics of both, and a trade-off can be made between the two by adjusting parameters. Usually is suitable for situations where there are a large number of features and there is a certain correlation between them.  
# 
#     Because Model_04 demonstrated the early sign of overfitting but the highest validation accuracy rate(0.8905). Although the model is a linear model(2 layers only), we still decided to introduce L1, L2, and L1+L2 regularization to check which of them works well on neutralized overfitting.
# 
#     4.3 Adding Dropout:  
#     Dropout is a technique commonly used to reduce overfitting, which prevents the model from being overly dependent on specific features of the training data set by randomly "turning off" a subset of neurons during training.  We can test how's the effect in Model_04 with only input and output layers (without hidden layers).
# 

# %% [markdown]
# # L1 regularization tuning summary:  
# Model_04-3(L1=0.00001) shows an overall improvement compared with Model_04, 04-1, and 04-2, even if it is still a little bit overfitting, we could use callback for later tuning.
# 
# ## Hence, we keep Model_04-3.

# %% [markdown]
# # Model_04-1:  
# layer=2  
# epochs=20  
# batch_size=256  
# learning rate=0.0005(Learning rate tuning winner)  
# L1=0.001(new)

# %%
# define model
model = tf.keras.models.Sequential([
tf.keras.layers.Dense(4, activation='relu', input_shape=(10000,),
                      kernel_regularizer=regularizers.l1(0.001)),
tf.keras.layers.Dense(1, activation='sigmoid')
])

# compile model
model.compile(optimizer=optimizers.RMSprop(learning_rate=0.0005),
              loss=losses.binary_crossentropy,
              metrics=['accuracy'])

# training and store training history
history = train_and_save_history(model, partial_x_train, partial_y_train, x_val, y_val, 20, 256, all_histories)

# evaluate model performance
results = model.evaluate(x_test, y_test)
print(f"Test Loss: {results[0]}, Test Accuracy: {results[1]}")

# plot last training loss and accuraccy
plot_history(all_histories[-1])

# %% [markdown]
# # Performance_Model_04-1

# %%
df = generate_performance_dataframe(model, history)
print(df)

clean_mem()

# %% [markdown]
# # Model_04-1 Summary:  
# The performance on Model_04:  
# Test Loss: 0.3078, Test Accuracy: 0.8783  
# Min.V.Loss 0.2725, Max.V.Acc 0.8906  
# 
# We can find that after introducing L1=0.001, all performance values on Model_04-1 are decreased. Even though the validation and test losses are on a downtrend, the validation accuracy value oscillates starting from epoch 11 till the end of the training(epoch 20), indicating an early sign of overfitting.

# %% [markdown]
# # Model_04-2:  
# layer=2  
# epochs=20  
# batch_size=256  
# learning rate=0.0005(Learning rate tuning winner)  
# L1=0.0005(new)

# %%
# Define model
model = tf.keras.models.Sequential([
tf.keras.layers.Dense(4, activation='relu', input_shape=(10000,),
                      kernel_regularizer=regularizers.l1(0.001)),
tf.keras.layers.Dense(1, activation='sigmoid')
])

# Complie model
model.compile(optimizer=optimizers.RMSprop(learning_rate=0.0005),
              loss=losses.binary_crossentropy,
              metrics=['accuracy'])

# Train model and store history
history = train_and_save_history(model, partial_x_train, partial_y_train, x_val, y_val, 20, 256, all_histories)

# Evaluate model performance
results = model.evaluate(x_test, y_test)
print(f"Test Loss: {results[0]}, Test Accuracy: {results[1]}")

# Plot both training and validation loss and accuracy of the last training
plot_history(all_histories[-1])

# %% [markdown]
# # Performance_Model_04-2

# %%
df = generate_performance_dataframe(model, history)
print(df)

clean_mem()

# %% [markdown]
# # Model_04-2 Summary:  
# We can find that after introducing L1=0.0005, all performance values are actually similar to Model_04-1. Even though the validation and test losses are on a downtrend, the validation accuracy value oscillates starting from epoch 4 till the end of the training(epoch 20), indicating an early sign of overfitting(earlier than Model_04-1).

# %% [markdown]
# # Model_04-3:  
# layer=2  
# epochs=20  
# batch_size=256  
# learning rate=0.0005(Learning rate tuning winner)  
# L1=0.00001(new)

# %%
# Define model
model = tf.keras.models.Sequential([
tf.keras.layers.Dense(4, activation='relu', input_shape=(10000,),
                      kernel_regularizer=regularizers.l1(0.00001)),
tf.keras.layers.Dense(1, activation='sigmoid')
])

# Complie model
model.compile(optimizer=optimizers.RMSprop(learning_rate=0.0005),
              loss=losses.binary_crossentropy,
              metrics=['accuracy'])

# Train model and store history
history = train_and_save_history(model, partial_x_train, partial_y_train, x_val, y_val, 20, 256, all_histories)

# Evaluate model performance
results = model.evaluate(x_test, y_test)
print(f"Test Loss: {results[0]}, Test Accuracy: {results[1]}")

# Plot both training and validation loss and accuracy of the last training
plot_history(all_histories[-1])

# %% [markdown]
# # Performance_Model_04-3

# %%
df = generate_performance_dataframe(model, history)
print(df)

clean_mem()

# %% [markdown]
# # Model_04-3 Summary:   
# The overall accuracy of Model04-3 (L1=0.0005) is slightly higher than Model_04 and the minimum validation loss is slightly higher than Model_04,  suggesting that L1=0.0005 may help the model generalize better to unseen data, so we can keep the Model_04-3.

# %% [markdown]
# # L2 regularization tuning summary:  
# Because the penalty is only added at training time, the loss for the model will be much higher at training than at test time. Comparing the overall performance among Model_04-4(L2=0.001), Model_04-5(L2=0.0005), and Model_04-6(L2=0.00001),  
#  ## we keep Model_04-6.

# %% [markdown]
# # Model_04-4:  
# layer=2  
# epochs=20  
# batch_size=256  
# learning rate=0.0005(Learning rate tuning winner)  
# L2=0.001(new)

# %%
# Define model
model = tf.keras.models.Sequential([
tf.keras.layers.Dense(4, activation='relu', input_shape=(10000,), kernel_regularizer=regularizers.l2(0.001)),
tf.keras.layers.Dense(1, activation='sigmoid', kernel_regularizer=regularizers.l2(0.001))
])

# Complie model
model.compile(optimizer=optimizers.RMSprop(learning_rate=0.0005),
              loss=losses.binary_crossentropy,
              metrics=['accuracy'])

# Train model and store history
history = train_and_save_history(model, partial_x_train, partial_y_train, x_val, y_val, 20, 256, all_histories)

# Evaluate model performance
results = model.evaluate(x_test, y_test)
print(f"Test Loss: {results[0]}, Test Accuracy: {results[1]}")

# Plot both training and validation loss and accuracy of the last training
plot_history(all_histories[-1])

# %% [markdown]
# # Performance_Model_04-4

# %%
df = generate_performance_dataframe(model, history)
print(df)

clean_mem()

# %% [markdown]
# # Model_04-4 Summary:  
# We can find that when L2=0.001, although the loss during the training and verification process is slightly higher, it shows slightly better accuracy on the test dataset and also reduces the problem of overfitting to a certain extent. This illustrates that L2 regularization is effective in improving the generalization ability of the model.

# %% [markdown]
# # Model_04-5:   
# layer=2  
# epochs=20  
# batch_size=256  
# learning rate=0.0005(Learning rate tuning winner)  
# L2=0.0005(new)

# %%
# Define model
model = tf.keras.models.Sequential([
tf.keras.layers.Dense(4, activation='relu', input_shape=(10000,), kernel_regularizer=regularizers.l2(0.0005)),
tf.keras.layers.Dense(1, activation='sigmoid', kernel_regularizer=regularizers.l2(0.0005))
])

# Complie model
model.compile(optimizer=optimizers.RMSprop(learning_rate=0.0005),
              loss=losses.binary_crossentropy,
              metrics=['accuracy'])

# Train model and store history
history = train_and_save_history(model, partial_x_train, partial_y_train, x_val, y_val, 20, 256, all_histories)

# Evaluate model performance
results = model.evaluate(x_test, y_test)
print(f"Test Loss: {results[0]}, Test Accuracy: {results[1]}")

# Plot both training and validation loss and accuracy of the last training
plot_history(all_histories[-1])

# %% [markdown]
# # Performance_Model_04-5

# %%
df = generate_performance_dataframe(model, history)
print(df)

clean_mem()

# %% [markdown]
# # Model_04-5 Summary:  
# Model_04-5 performs better in the early stages of training but is slightly worse than Model_04 in final test accuracy. Moreover, Model_04 shows relatively stable performance throughout the training process, especially in the validation phase.

# %% [markdown]
# # Model_04-6:   
# layer=2  
# epochs=20  
# batch_size=256  
# learning rate=0.0005(Learning rate tuning winner)  
# L2=0.00001(new)

# %%
# Define model
model = tf.keras.models.Sequential([
tf.keras.layers.Dense(4, activation='relu', input_shape=(10000,), kernel_regularizer=regularizers.l2(0.00001)),
tf.keras.layers.Dense(1, activation='sigmoid', kernel_regularizer=regularizers.l2(0.00001))
])

# Complie model
model.compile(optimizer=optimizers.RMSprop(learning_rate=0.0005),
              loss=losses.binary_crossentropy,
              metrics=['accuracy'])

# Train model and store history
history = train_and_save_history(model, partial_x_train, partial_y_train, x_val, y_val, 20, 256, all_histories)

# Evaluate model performance
results = model.evaluate(x_test, y_test)
print(f"Test Loss: {results[0]}, Test Accuracy: {results[1]}")

# Plot both training and validation loss and accuracy of the last training
plot_history(all_histories[-1])

# %% [markdown]
# # Performance_Model_04-6

# %%
df = generate_performance_dataframe(model, history)
print(df)

clean_mem()

# %% [markdown]
# # Model_04-6 Summary:  
# The loss values ​​of Model_04-6 gradually decrease during the training process, but the loss value decreases more smoothly. Furthermore, Model_04-6 validation accuracy value is slightly higher than Model_04. On the training set, the accuracy of training result 2 (88.30%) is slightly higher than training result 1 (87.84%), suggesting the idea that L2 regularization helps control the complexity of the model to avoid overfitting. Therefore, Model_04-6 will be kept.
# 

# %% [markdown]
# # L1+L2 regularization(Model_04-7) tuning summary:  
# The overall Moel_04-7 performance on test loss, test accuracy, and maximum validation accuracy is slightly better than Model_04-6, even though both models show the signs of overfitting. Therefore,  
# ## we use Model_04-7 for tuning dropout.  

# %% [markdown]
# # Model_04-7:  
# layer=2  
# epochs=20  
# batch_size=256  
# learning rate=0.0005(Learning rate tuning winner)  
# L1=0.00001(from Model_04-3)  
# L2=0.00001(from Model_04-6)

# %%
# Define model
model = tf.keras.models.Sequential([
tf.keras.layers.Dense(4, activation='relu', input_shape=(10000,), kernel_regularizer=regularizers.l2(0.00001)),
tf.keras.layers.Dense(1, activation='sigmoid', kernel_regularizer=regularizers.l1_l2(l1=0.00001, l2=0.00001))
])

# Complie model
model.compile(optimizer=optimizers.RMSprop(learning_rate=0.0005),
              loss=losses.binary_crossentropy,
              metrics=['accuracy'])

# Train model and store history
history = train_and_save_history(model, partial_x_train, partial_y_train, x_val, y_val, 20, 256, all_histories)

# Evaluate model performance
results = model.evaluate(x_test, y_test)
print(f"Test Loss: {results[0]}, Test Accuracy: {results[1]}")

# Plot both training and validation loss and accuracy of the last training
plot_history(all_histories[-1])

# %% [markdown]
# # Performance_Model_04-7

# %%
df = generate_performance_dataframe(model, history)
print(df)

clean_mem()

# %% [markdown]
#  # Dropout tuning summary:  
# Four Models of results as below and all of them are showing overfitting:
# 
# Model_04-8, dropout=0.5:  
# Test Loss: 0.2956584393978119, Test Accuracy: 0.8817600011825562  
# Min.V.Loss: 0.280491,  Max.V.Acc:0.8893  
# 
# Model_04-9, dropout=0.1:  
# Test Loss: 0.2943788170814514, Test Accuracy: 0.8817200064659119
# Min.V.Loss: 0.271153,  Max.V.Acc:0.8915  
# 
# Model_04-10, dropout=0.05:  
# Test Loss: 0.29925766587257385, Test Accuracy: 0.8803600072860718
# Min.V.Loss: 0.270322,  Max.V.Acc:0.8916  
# 
# Model_04-11, dropout=0.01:  
# Test Loss: 0.2997041642665863, Test Accuracy: 0.8817600011825562
# Min.V.Loss: 0.272967,  Max.V.Acc:0.8900  
# 
# Apparently, all values are close, so we choose based on the maximum validation accuracy, 0.8916.  
# 
# ## Hence, Model_4-10 is our final model.  
# However, Model_4-10 also shows signs of overfitting, so we will further use EarlyStopping based on Model_4-10 as Model_4-12.

# %% [markdown]
# # Model_04-8:  
# layer=3  
# epochs=20  
# batch_size=256  
# learning rate=0.0005(Learning rate tuning winner)  
# L1=0.00001(from Model_04-7)  
# L2=0.00001(from Model_04-7)  
# dropout=0.5(new)

# %%
# Define model
model = tf.keras.models.Sequential([
tf.keras.layers.Dense(4, activation='relu', input_shape=(10000,), kernel_regularizer=regularizers.l2(0.00001)),
tf.keras.layers.Dropout(0.5),
tf.keras.layers.Dense(1, activation='sigmoid', kernel_regularizer=regularizers.l1_l2(l1=0.00001, l2=0.00001))
])

# Complie model
model.compile(optimizer=optimizers.RMSprop(learning_rate=0.0005),
              loss=losses.binary_crossentropy,
              metrics=['accuracy'])

# Train model and store history
history = train_and_save_history(model, partial_x_train, partial_y_train, x_val, y_val, 20, 256, all_histories)

# Evaluate model performance
results = model.evaluate(x_test, y_test)
print(f"Test Loss: {results[0]}, Test Accuracy: {results[1]}")

# Plot both training and validation loss and accuracy of the last training
plot_history(all_histories[-1])

# %% [markdown]
# # Performance_Model_04-8

# %%
df = generate_performance_dataframe(model, history)
print(df)

clean_mem()

# %% [markdown]
# # Model_04-9:  
# layer=3  
# epochs=20  
# batch_size=256  
# learning rate=0.0005(Learning rate tuning winner)  
# L1=0.00001(from Model_04-7)  
# L2=0.00001(from Model_04-7)  
# dropout=0.1(new)

# %%
# Define model
model = tf.keras.models.Sequential([
tf.keras.layers.Dense(4, activation='relu', input_shape=(10000,), kernel_regularizer=regularizers.l2(0.00001)),
tf.keras.layers.Dropout(0.1),
tf.keras.layers.Dense(1, activation='sigmoid', kernel_regularizer=regularizers.l1_l2(l1=0.00001, l2=0.00001))
])

# Complie model
model.compile(optimizer=optimizers.RMSprop(learning_rate=0.0005),
              loss=losses.binary_crossentropy,
              metrics=['accuracy'])

# Train model and store history
history = train_and_save_history(model, partial_x_train, partial_y_train, x_val, y_val, 20, 256, all_histories)

# Evaluate model performance
results = model.evaluate(x_test, y_test)
print(f"Test Loss: {results[0]}, Test Accuracy: {results[1]}")

# Plot both training and validation loss and accuracy of the last training
plot_history(all_histories[-1])

# %% [markdown]
# # Performance_Model_04-9

# %%
df = generate_performance_dataframe(model, history)
print(df)

clean_mem()

# %% [markdown]
# # Model_04-10:  
# layer=3  
# epochs=20  
# batch_size=256  
# learning rate=0.0005(Learning rate tuning winner)  
# L1=0.00001(from Model_04-7)  
# L2=0.00001(from Model_04-7)  
# dropout=0.05(new)

# %%
# Define model
model = tf.keras.models.Sequential([
tf.keras.layers.Dense(4, activation='relu', input_shape=(10000,), kernel_regularizer=regularizers.l2(0.00001)),
tf.keras.layers.Dropout(0.05),
tf.keras.layers.Dense(1, activation='sigmoid', kernel_regularizer=regularizers.l1_l2(l1=0.00001, l2=0.00001))
])

# Complie model
model.compile(optimizer=optimizers.RMSprop(learning_rate=0.0005),
              loss=losses.binary_crossentropy,
              metrics=['accuracy'])

# Train model and store history
history = train_and_save_history(model, partial_x_train, partial_y_train, x_val, y_val, 20, 256, all_histories)

# Evaluate model performance
results = model.evaluate(x_test, y_test)
print(f"Test Loss: {results[0]}, Test Accuracy: {results[1]}")

# Plot both training and validation loss and accuracy of the last training
plot_history(all_histories[-1])

# %% [markdown]
# # Performance_Model_04-10

# %%
df = generate_performance_dataframe(model, history)
print(df)

clean_mem()

# %% [markdown]
# # Model_04-11:    
# layer=3  
# epochs=20  
# batch_size=256  
# learning rate=0.0005(Learning rate tuning winner)  
# L1=0.00001(from Model_04-7)  
# L2=0.00001(from Model_04-7)  
# dropout=0.01(new)

# %%
# Define model
model = tf.keras.models.Sequential([
tf.keras.layers.Dense(4, activation='relu', input_shape=(10000,), kernel_regularizer=regularizers.l2(0.00001)),
tf.keras.layers.Dropout(0.01),
tf.keras.layers.Dense(1, activation='sigmoid', kernel_regularizer=regularizers.l1_l2(l1=0.00001, l2=0.00001))
])

# Complie model
model.compile(optimizer=optimizers.RMSprop(learning_rate=0.0005),
              loss=losses.binary_crossentropy,
              metrics=['accuracy'])

# Train model and store history
history = train_and_save_history(model, partial_x_train, partial_y_train, x_val, y_val, 20, 256, all_histories)

# Evaluate model performance
results = model.evaluate(x_test, y_test)
print(f"Test Loss: {results[0]}, Test Accuracy: {results[1]}")

# Plot both training and validation loss and accuracy of the last training
plot_history(all_histories[-1])

# %% [markdown]
# # Performance_Model_04-11

# %%
df = generate_performance_dataframe(model, history)
print(df)

clean_mem()

# %% [markdown]
# # Model_4-12(earlystopping)
# layer=3  
# epochs=20  
# batch_size=256  
# learning rate=0.0005(Learning rate tuning winner)  
# L1=0.00001(from Model_04-7)  
# L2=0.00001(from Model_04-7)  
# dropout=0.05(from Modeul_04-10)  
# earlystopping: patience=4
# 

# %%
# idea comes from Jérémie walk through about IMDB dataset, so I ask ChatGPT Plugins if there's any automation early stopping test for machine learning
# codes are generated by ChatGPT Plugins, 14/11
# prompt: copy and past one of the model code and ask it to use automation early stopping.

# Define model
model = tf.keras.models.Sequential([
tf.keras.layers.Dense(4, activation='relu', input_shape=(10000,), kernel_regularizer=regularizers.l2(0.00001)),
tf.keras.layers.Dropout(0.05),
tf.keras.layers.Dense(1, activation='sigmoid', kernel_regularizer=regularizers.l1_l2(l1=0.00001, l2=0.00001))
])

# Complie model
model.compile(optimizer=optimizers.RMSprop(learning_rate=0.0005),
              loss=losses.binary_crossentropy,
              metrics=['accuracy'])

# Set EarlyStopping callback
early_stopping = EarlyStopping(monitor='val_accuracy', patience=3, restore_best_weights=True)

# Train the model and save history
# Make sure the return value is assigned to the history variable and add an early_stopping callback
history = train_and_save_history(model, partial_x_train, partial_y_train, x_val, y_val, 20, 256, all_histories, callbacks=[early_stopping])

# Evaluate model performance
results = model.evaluate(x_test, y_test)
print(f"Test Loss: {results[0]}, Test Accuracy: {results[1]}")

# Plot both training and validation loss and accuracy of the last training
plot_history(all_histories[-1])

# %%
df = generate_performance_dataframe(model, history)
print(df)

clean_mem()

# %% [markdown]
# # Final Model

# %%
# Define model without validation
def train_and_save_history_final(model, x_train, y_train, epochs, batch_size, all_histories, callbacks=None):
    history = model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size, callbacks=callbacks)
        all_histories.append(history)
    return history

model = tf.keras.models.Sequential([
tf.keras.layers.Dense(4, activation='relu', input_shape=(10000,), kernel_regularizer=regularizers.l2(0.00001)),
tf.keras.layers.Dropout(0.05),
tf.keras.layers.Dense(1, activation='sigmoid', kernel_regularizer=regularizers.l1_l2(l1=0.00001, l2=0.00001))
])

# Complie model
model.compile(optimizer=optimizers.RMSprop(learning_rate=0.0005),
              loss=losses.binary_crossentropy,
              metrics=['accuracy'])

# Train the model and save history
# Make sure the return value is assigned to the history variable and add an early_stopping callback
history = train_and_save_history_final(model, x_train, y_train, epochs=13, batch_size=256, all_histories=all_histories)



# %%
# Evaluate model performance
results = model.evaluate(x_test, y_test)
print(f"Test Loss: {results[0]}, Test Accuracy: {results[1]}")

# %%
clean_mem()

# %% [markdown]
# # Conclusion:  
# The final model result on test set accuracy indicates that the loss is steady and gradually going down, and the accuracy reaches a good value(0.9483) among all training models. From this research, we can refer to the "Learning Rate" as the key point, at least for the RMSProp optimizer. I'll say we are wise(lucky?) that we are so close to the best result at the first step since we followed the DLWP at the very beginning. However, the performance on the test set doesn't perform well at the same level. It is worth it to discuss the possible reasons.  
# 
# ###From the dataset accurate aspect:
# 1. The IMDB datasets had been randomized when it was released, so this is not the whether the dataset split randomly problem.
# 
# 2. The datasets need more preprocess. According to the post "IMDB sentiment Classifier ~ 97% accuracy model" on Kaggle, perhaps using glove embeddings and LSTM layers would be helpful, but that's not cover in our learning scope for CW 1.
# 
# 3. Data leakage. Our model has been spliti into 3 different parts correctly and independently, hence we could exclude this reason.
# 
# ###From model aspect:
# 1. The model reaches its limit. Because our model is relatively simple and contains only 2 layers and 4 units, maybe it is unable to learn from or perform a more complicated classification mission.
# 
# 2. Base on Model_04-12, choosing a different epoch on final test dasets. We can found that the early sign of overfitting on epoch 4, not epoch 13. Moreover, the maximum validation accuracy happened at epoch 10, perhpas this is also a good point.
# 
# Therefore, maybe learning rate the key of machine learning, but the dataset could be much better preprocessed before we start to train a model. Furtheromre, if we find an ideal learning rate for our model, maybe add more layers and units could help improve the performance.

# %% [markdown]
# 
# 
# ---
# 
# 
# 
# ---
# 
# 

# %% [markdown]
# # Uphold the fearless spirit of a brave warrior, just as we defeat the chilling Friday morning, we shall do more experiments.

# %% [markdown]
# # 11. Further test

# %% [markdown]
# ## Model_04
# ### (For reference and comparison, so we don't have to roll allover to the top)

# %%
# Define model
model = tf.keras.models.Sequential([
tf.keras.layers.Dense(4, activation='relu', input_shape=(10000,)),
tf.keras.layers.Dense(1, activation='sigmoid')
])

# Complie model
model.compile(optimizer=optimizers.RMSprop(learning_rate=0.0005),
              loss=losses.binary_crossentropy,
              metrics=['accuracy'])

# Train model and store history
history = train_and_save_history(model, partial_x_train, partial_y_train, x_val, y_val, 20, 256, all_histories)

# Evaluate model performance
results = model.evaluate(x_test, y_test)
print(f"Test Loss: {results[0]}, Test Accuracy: {results[1]}")

# Plot both training and validation loss and accuracy of the last training
plot_history(all_histories[-1])

df = generate_performance_dataframe(model, history)
print(df)

clean_mem()

# %% [markdown]
# ## Performance_Model_4

# %% [markdown]
# Min.V.Loss: 0.272541   
# 
# Max.V.Acc: 0.8906

# %% [markdown]
# ## Model_04-10  
# ### (For reference and comparison, so we don't have to roll allover to the top)

# %%
# Define model
model = tf.keras.models.Sequential([
tf.keras.layers.Dense(4, activation='relu', input_shape=(10000,), kernel_regularizer=regularizers.l2(0.00001)),
tf.keras.layers.Dropout(0.05),
tf.keras.layers.Dense(1, activation='sigmoid', kernel_regularizer=regularizers.l1_l2(l1=0.00001, l2=0.00001))
])

# Complie model
model.compile(optimizer=optimizers.RMSprop(learning_rate=0.0005),
              loss=losses.binary_crossentropy,
              metrics=['accuracy'])

# Train model and store history
history = train_and_save_history(model, partial_x_train, partial_y_train, x_val, y_val, 20, 256, all_histories)

# Evaluate model performance
results = model.evaluate(x_test, y_test)
print(f"Test Loss: {results[0]}, Test Accuracy: {results[1]}")

# Plot both training and validation loss and accuracy of the last training
plot_history(all_histories[-1])

df = generate_performance_dataframe(model, history)
print(df)

clean_mem()

# %% [markdown]
# ## Performance_Model_4-10

# %% [markdown]
# Min.V.Loss: 0.270322  
# 
# Max.V.Acc: 0.8916
#          

# %% [markdown]
# 
# 
# ---
# 
# 
# # MSE loss function summary:  
# MSE(Mean Squared Error) is a commonly used loss function, usually and especially in regression problems. We can easily find that the loss is significantly decreased, which fits the feature of MSE. But the overfitting still exist.

# %% [markdown]
# ## Model_04_MSE

# %%
# Define model
model = tf.keras.models.Sequential([
tf.keras.layers.Dense(4, activation='relu', input_shape=(10000,)),
tf.keras.layers.Dense(1, activation='sigmoid')
])

# Complie model
model.compile(optimizer=optimizers.RMSprop(learning_rate=0.0005),
              loss='mean_squared_error',  # Changed to MSE
              metrics=['accuracy'])

# Train model and store history
history = train_and_save_history(model, partial_x_train, partial_y_train, x_val, y_val, 20, 256, all_histories)

# Evaluate model performance
results = model.evaluate(x_test, y_test)
print(f"Test Loss: {results[0]}, Test Accuracy: {results[1]}")

# Plot both training and validation loss and accuracy of the last training
plot_history(all_histories[-1])

df = generate_performance_dataframe(model, history)
print(df)

clean_mem()

# %% [markdown]
# ## Model_4-10_MSE

# %%
# Define model
model = tf.keras.models.Sequential([
tf.keras.layers.Dense(4, activation='relu', input_shape=(10000,), kernel_regularizer=regularizers.l2(0.00001)),
tf.keras.layers.Dropout(0.05),
tf.keras.layers.Dense(1, activation='sigmoid', kernel_regularizer=regularizers.l1_l2(l1=0.00001, l2=0.00001))
])

# Complie model
model.compile(optimizer=optimizers.RMSprop(learning_rate=0.0005),
              loss='mean_squared_error',  # Changed to MSE
              metrics=['accuracy'])

# Train model and store history
history = train_and_save_history(model, partial_x_train, partial_y_train, x_val, y_val, 20, 256, all_histories)

# Evaluate model performance
results = model.evaluate(x_test, y_test)
print(f"Test Loss: {results[0]}, Test Accuracy: {results[1]}")

# Plot both training and validation loss and accuracy of the last training
plot_history(all_histories[-1])

df = generate_performance_dataframe(model, history)
print(df)

clean_mem()

# %% [markdown]
# 
# 
# ---
# 
# 
# # Tanh activation summary:  
# ### (Manual edit content, reference from ChatGPT Plugins)
# The use cases of tanh are often in "hidden layers" of a neural network. It's particularly popular in tasks where normalization of input is beneficial, and in networks where negative output values can be meaningful. Moreover, the drawbacks include the vanishing gradient problem in deep networks. Due to the tanh function outputs values in the range of -1 to 1, meaning the output value could close toan  0. We can find that the loss decreases and the validation accuracy is higher than ReLU, so for simple networks, using tanh as avtivation could be an option, even though the overfitting occurs early.

# %% [markdown]
# ## Model_04_tanh

# %%
# Define model
model = tf.keras.models.Sequential([
tf.keras.layers.Dense(4, activation='tanh', input_shape=(10000,)),  # Changed to 'tanh'
tf.keras.layers.Dense(1, activation='sigmoid')
])

# Complie model
model.compile(optimizer=optimizers.RMSprop(learning_rate=0.0005),
              loss=losses.binary_crossentropy,
              metrics=['accuracy'])

# Train model and store history
history = train_and_save_history(model, partial_x_train, partial_y_train, x_val, y_val, 20, 256, all_histories)

# Evaluate model performance
results = model.evaluate(x_test, y_test)
print(f"Test Loss: {results[0]}, Test Accuracy: {results[1]}")

# Plot both training and validation loss and accuracy of the last training
plot_history(all_histories[-1])

df = generate_performance_dataframe(model, history)
print(df)

clean_mem()

# %% [markdown]
# ## Model_4-10_tanh

# %%
# Define model
model = tf.keras.models.Sequential([
tf.keras.layers.Dense(4, activation='tanh', input_shape=(10000,), kernel_regularizer=regularizers.l2(0.00001)), # Changed to 'tanh'
tf.keras.layers.Dropout(0.05),
tf.keras.layers.Dense(1, activation='sigmoid', kernel_regularizer=regularizers.l1_l2(l1=0.00001, l2=0.00001))
])

# Complie model
model.compile(optimizer=optimizers.RMSprop(learning_rate=0.0005),
              loss='mean_squared_error',  # Changed to MSE
              metrics=['accuracy'])

# Train model and store history
history = train_and_save_history(model, partial_x_train, partial_y_train, x_val, y_val, 20, 256, all_histories)

# Evaluate model performance
results = model.evaluate(x_test, y_test)
print(f"Test Loss: {results[0]}, Test Accuracy: {results[1]}")

# Plot both training and validation loss and accuracy of the last training
plot_history(all_histories[-1])

df = generate_performance_dataframe(model, history)
print(df)

clean_mem()

# %% [markdown]
# 
# 
# ---
# 
# 
# # Momentum summary:  
# Although the training performance is good, the validation loss gradually increases starting from epoch 2, while the validation accuracy drops slightly after peaking at epoch 2. A classic sign of overfitting, meaning that the model performed so well on the training data that it loses its ability to generalize to unseen data.

# %% [markdown]
# ## Model_04_momentum(0.9)
# 

# %%
# Define model
model = tf.keras.models.Sequential([
tf.keras.layers.Dense(4, activation='relu', input_shape=(10000,)),
tf.keras.layers.Dense(1, activation='sigmoid')
])

# Complie model
model.compile(optimizer=optimizers.RMSprop(learning_rate=0.0005, momentum=0.9),
              loss=losses.binary_crossentropy,
              metrics=['accuracy'])

# Train model and store history
history = train_and_save_history(model, partial_x_train, partial_y_train, x_val, y_val, 20, 256, all_histories)

# Evaluate model performance
results = model.evaluate(x_test, y_test)
print(f"Test Loss: {results[0]}, Test Accuracy: {results[1]}")

# Plot both training and validation loss and accuracy of the last training
plot_history(all_histories[-1])

df = generate_performance_dataframe(model, history)
print(df)

clean_mem()

# %%
# Define model
model = tf.keras.models.Sequential([
tf.keras.layers.Dense(4, activation='relu', input_shape=(10000,), kernel_regularizer=regularizers.l2(0.00001)),
tf.keras.layers.Dropout(0.05),
tf.keras.layers.Dense(1, activation='sigmoid', kernel_regularizer=regularizers.l1_l2(l1=0.00001, l2=0.00001))
])

# Complie model
model.compile(optimizer=optimizers.RMSprop(learning_rate=0.0005),
              loss=losses.binary_crossentropy,
              metrics=['accuracy'])

# Train model and store history
history = train_and_save_history(model, partial_x_train, partial_y_train, x_val, y_val, 20, 256, all_histories)

# Evaluate model performance
results = model.evaluate(x_test, y_test)
print(f"Test Loss: {results[0]}, Test Accuracy: {results[1]}")

# Plot both training and validation loss and accuracy of the last training
plot_history(all_histories[-1])

df = generate_performance_dataframe(model, history)
print(df)

clean_mem()

# %% [markdown]
# ## Model_4-10_momentum(0.5)

# %%
# Define model
model = tf.keras.models.Sequential([
tf.keras.layers.Dense(4, activation='relu', input_shape=(10000,), kernel_regularizer=regularizers.l2(0.00001)),
tf.keras.layers.Dropout(0.05),
tf.keras.layers.Dense(1, activation='sigmoid', kernel_regularizer=regularizers.l1_l2(l1=0.00001, l2=0.00001))
])

# Complie model
model.compile(optimizer=optimizers.RMSprop(learning_rate=0.0005, momentum=0.5),
              loss=losses.binary_crossentropy,
              metrics=['accuracy'])

# Train model and store history
history = train_and_save_history(model, partial_x_train, partial_y_train, x_val, y_val, 20, 256, all_histories)

# Evaluate model performance
results = model.evaluate(x_test, y_test)
print(f"Test Loss: {results[0]}, Test Accuracy: {results[1]}")

# Plot both training and validation loss and accuracy of the last training
plot_history(all_histories[-1])

df = generate_performance_dataframe(model, history)
print(df)

clean_mem()

# %% [markdown]
# 
# 
# ---
# 
# # Adam summary:  
# ### (Manual edit content, reference from ChatGPT Plugins)
# The Adam optimizer adjusts the learning rate of each parameter based on the moving average of the square of the gradient of each parameter. This means it provides a custom learning rate for each parameter, making the optimization process more efficient and effective. Furthermore, due to its adaptive learning rate feature, Adam performs well on many different types of deep learning models and different data sets, especially in applications with large data sets and high-dimensional spaces. While compared to other optimization algorithms, Adam is less sensitive to the choice of initial learning rate and other hyperparameters, which makes it easier to configure in practice. However, overfitting still happened and it happened in early epoch.

# %% [markdown]
# ## Model_04_Adam

# %%
# Define model
model = tf.keras.models.Sequential([
tf.keras.layers.Dense(4, activation='relu', input_shape=(10000,)),
tf.keras.layers.Dense(1, activation='sigmoid')
])

# Complie model
model.compile(optimizer='adam',
              loss=losses.binary_crossentropy,
              metrics=['accuracy'])

# Train model and store history
history = train_and_save_history(model, partial_x_train, partial_y_train, x_val, y_val, 20, 256, all_histories)

# Evaluate model performance
results = model.evaluate(x_test, y_test)
print(f"Test Loss: {results[0]}, Test Accuracy: {results[1]}")

# Plot both training and validation loss and accuracy of the last training
plot_history(all_histories[-1])

df = generate_performance_dataframe(model, history)
print(df)

clean_mem()

# %% [markdown]
# ## Model_04-10_Adam

# %%
# Define model
model = tf.keras.models.Sequential([
tf.keras.layers.Dense(4, activation='relu', input_shape=(10000,), kernel_regularizer=regularizers.l2(0.00001)),
tf.keras.layers.Dropout(0.05),
tf.keras.layers.Dense(1, activation='sigmoid', kernel_regularizer=regularizers.l1_l2(l1=0.00001, l2=0.00001))
])

# Complie model
model.compile(optimizer='adam',
              loss=losses.binary_crossentropy,
              metrics=['accuracy'])

# Train model and store history
history = train_and_save_history(model, partial_x_train, partial_y_train, x_val, y_val, 20, 256, all_histories)

# Evaluate model performance
results = model.evaluate(x_test, y_test)
print(f"Test Loss: {results[0]}, Test Accuracy: {results[1]}")

# Plot both training and validation loss and accuracy of the last training
plot_history(all_histories[-1])

df = generate_performance_dataframe(model, history)
print(df)

clean_mem()

# %% [markdown]
# # Summary of further challenges against overfitting:  
# Unfortunately, after our hard trying, both overfitting and overall model performance weren't improved in any way. We can basically conclude that for a simple network model, learning accounts for an outstanding role.  
# 
# # Therefore, in order to see how those hyperparameters affect the performance, we are further testing on more layers and units. Based on Model_4-10, since Adam is a popular optimizer in deep learning and it combines Momentum and RMSprop stochastic gradient descent methods, we change the optimizer from RMSProp to Adam.

# %% [markdown]
# 
# 
# ---
# 
# 
# 
# ---
# 
# 

# %% [markdown]
# # Model_4-10_Adam_01  
# 
# layer=4(new)  
# epochs=20  
# batch_size=256  
# optimizer=adam(new)  
# hidden layer activation=relu(new)  
# 
# 

# %%
# Define model
model = tf.keras.models.Sequential([
tf.keras.layers.Dense(4, activation='relu', input_shape=(10000,)),
tf.keras.layers.Dense(4, activation='relu',),
tf.keras.layers.Dense(1, activation='sigmoid')
])

# Complie model
model.compile(optimizer='adam',
              loss=losses.binary_crossentropy,
              metrics=['accuracy'])

# Train model and store history
history = train_and_save_history(model, partial_x_train, partial_y_train, x_val, y_val, 20, 256, all_histories)

# Evaluate model performance
results = model.evaluate(x_test, y_test)
print(f"Test Loss: {results[0]}, Test Accuracy: {results[1]}")

# Plot both training and validation loss and accuracy of the last training
plot_history(all_histories[-1])

df = generate_performance_dataframe(model, history)
print(df)

clean_mem()

# %% [markdown]
# # Model_4-10_Adam_02  
# 
# layer=4(new)  
# epochs=20  
# batch_size=256  
# optimizer=adam(new)  
# hidden layer activation=tanh(new)  

# %%
# Define model
model = tf.keras.models.Sequential([
tf.keras.layers.Dense(4, activation='relu', input_shape=(10000,)),
tf.keras.layers.Dense(4, activation='tanh'),
tf.keras.layers.Dense(1, activation='sigmoid')
])

# Complie model
model.compile(optimizer='adam',
              loss=losses.binary_crossentropy,
              metrics=['accuracy'])

# Train model and store history
history = train_and_save_history(model, partial_x_train, partial_y_train, x_val, y_val, 20, 256, all_histories)

# Evaluate model performance
results = model.evaluate(x_test, y_test)
print(f"Test Loss: {results[0]}, Test Accuracy: {results[1]}")

# Plot both training and validation loss and accuracy of the last training
plot_history(all_histories[-1])

df = generate_performance_dataframe(model, history)
print(df)

clean_mem()

# %% [markdown]
# # Model_4-10_Adam_02 Summary:  
# Reaching the highest validation accuracy of 0.8928, but the loss is quite high.

# %% [markdown]
# 
# 
# 
# ---
# 
# 
# ---
# 
# 
# 
# # Try and think about how you would go about automating hyperparameter search (= making your life easier, so that the search is done by a system rather than manually)  
# 
# ## We try "Grid Search" here, but we find there are some best practice in Ch.13 of DLWP. Grid Search could be a solution, but it depends on the memory on the running device. Now we can have a concept about why memory administration is important and how important it is - we are not machines, we are human beings.

# %% [markdown]
# # Model_4-10_Adam_03_Grid Search  
# ### Codes generate by ChatGPT(ver. Plugins)
# ### prompt: given Model_4, ask GPT give the code of grid search
# 
# fixed:  
# batch_sizes = [256]  
# learning_rates = [0.0005]  
# epoch=10
# 
# grid search in:
# layer=4(new)  
# optimizer=adam(new)  
# hidden layer activation=tanh(new)  
# l2_regs = [0.001, 0.0001]

# %%
# idea comes from Jérémie walk through about IMDB dataset, so I ask ChatGPT Plugins if there's any automation test for machine learning
# codes are generated by ChatGPT Plugins, 15/11
# prompt: copy and past one of the model code and ask it to generate grid search for machine learning
# it originally defined batch sizes as 64, 128, 256, 512, and 1024, learning reate 0.01-0.0001, but I manually change the batch sizes(the colab memory limit) and learning rates(proven 0.0005 is the best)

# Define grid
batch_sizes = [256]
learning_rates = [0.0005]
l2_regs = [0.001, 0.0001]

# Save the best models and performance
best_accuracy = 0
best_params = {}

# Loop grid search
for batch_size in batch_sizes:
    for lr in learning_rates:
        for l2_reg in l2_regs:
            print(f"Training with batch size: {batch_size}, learning rate: {lr}, L2 regularization: {l2_reg}")

            # Define model
            model = tf.keras.models.Sequential([
                tf.keras.layers.Dense(4, activation='relu', input_shape=(10000,), kernel_regularizer=regularizers.l2(l2_reg)),
                tf.keras.layers.Dense(4, activation='tanh', kernel_regularizer=regularizers.l2(l2_reg)),
                tf.keras.layers.Dense(1, activation='sigmoid')
            ])

            # Complie model
            optimizer = optimizers.Adam(learning_rate=lr)
            model.compile(optimizer=optimizer,
                          loss=losses.binary_crossentropy,
                          metrics=['accuracy'])

            # Train model and store history
            history = model.fit(partial_x_train, partial_y_train,
                                epochs=10,
                                batch_size=batch_size,
                                validation_data=(x_val, y_val))

            # Evaluate model performance
            results = model.evaluate(x_test, y_test)
            print(f"Test Loss: {results[0]}, Test Accuracy: {results[1]}")

            # Update best model
            if results[1] > best_accuracy:
                best_accuracy = results[1]
                best_params = {'batch_size': batch_size, 'learning_rate': lr, 'l2_reg': l2_reg}

            # Clean memory
            tf.keras.backend.clear_session()

print(f"Best Accuracy: {best_accuracy}")
print(f"Best Parameters: {best_params}")

# %% [markdown]
# 
# 
# ---
# 
# 
# 
# ---
# # 12. Postscript
# 
# Hyperparameter fine-tuning is a tedious task. Throughout this whole process, I felt like it could definitely be automated because most of the time we were changing some hyperparameters. This is the advantage of computers, as smart working will always be important, just as Robotic Process Automation (RPA) becomes popular in industry. Moreover, I have some experience in managing Facebook and Google ads. I feel that every time we set the conditions for the advertising management platform, we adjust some parameters, and the algorithm will automatically find the best offer. That’s the end of CW 1, but I found that there is more NLP stuff in Ch.11 of DLWP, so I’m really looking forward to it.

# %% [markdown]
# # 13. References  
# 1. Chollet F., 2021, Ch.1-6, Deep Learning with Python, Second Edition, Manning
# 2. Wenger J., Week1-5, 2023-2024, Lecture and Lab materials, IS53024B/S/IS71039B/A: ARTIFICIAL INTELLIGENCE (2023-24), Goldsmiths, UoL
# 3. ChatGPT Plugins, 6/11-15/11, 2023
# 4. Pykes K., 2023, Fighting Overfitting With L1 or L2 Regularization: Which One Is Better? Asseciable: https://neptune.ai/blog/fighting-overfitting-with-l1-or-l2-regularization#:~:text=L1%20regularization%20is%20more%20robust,the%20cost%20only%20increases%20linearly.


