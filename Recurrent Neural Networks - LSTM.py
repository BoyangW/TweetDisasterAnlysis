# -*- coding: utf-8 -*-
"""
This file consists of LSTM recurrent neural networks and  evaluation on each 
combination with its insights. The data used is Disaster Tweets from Kaggle 
Competition.
"""

import gc
import string
import operator
from collections import defaultdict
from wordcloud import STOPWORDS
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit
from sklearn.metrics import precision_score, recall_score, f1_score
from tensorflow import keras
from tensorflow.keras.optimizers import SGD, Adam
from tensorflow.keras.layers import Dense, Input, Dropout, GlobalAveragePooling1D
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, Callback
from sklearn import model_selection
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline


import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Conv2D, MaxPooling2D, UpSampling2D
from tensorflow.keras.layers import Dropout, Input, Embedding, LSTM
from tensorflow.keras.models import Model,Sequential
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing import sequence
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import EarlyStopping
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
import string
import re

#Read cleaned dataset
df_clean = pd.read_csv('data/clean.csv', dtype={'id': np.int16, 'target': np.int8})

#LSTM
# The maximum number of words to be used. (most frequent)
MAX_NB_WORDS = 30000
# Max number of words in each tweet
MAX_SEQUENCE_LENGTH = 1000
# Number of embedding dimensions (projection)
EMBEDDING_DIM = 100

# Additional tokenization 
tokenizer = Tokenizer(num_words=MAX_NB_WORDS, filters='!"#$%&()*+,-./:;<=>?@[\]^_`{|}~', lower=True)
tokenizer.fit_on_texts(df_clean['text'].values)
word_index = tokenizer.word_index
print('Found %s unique tokens.' % len(word_index))

# Parse reviews' text to embedding vectors 
X = tokenizer.texts_to_sequences(df_clean['text'].values)
X = sequence.pad_sequences(X, maxlen=MAX_SEQUENCE_LENGTH)
Y = df_clean['target'].values

print('Shape of data tensor:', X.shape)
print('Shape of label tensor:', Y.shape)

# Split training and testing set
X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size = 0.10, random_state = 42)
print(X_train.shape,Y_train.shape)
print(X_test.shape,Y_test.shape)

# set up RNN structure
def SetupModel():    
    model = Sequential()
    model.add(Embedding(MAX_NB_WORDS, EMBEDDING_DIM, input_length=X_train.shape[1]))
    model.add(Dropout(0.2))
    model.add(LSTM(100, dropout=0.2, recurrent_dropout=0.2)) #LSTM layer
    model.add(Dense(1, activation='sigmoid'))
    return(model)

# set up number of epoches batch size
epochs = 5
batch_size = 32

SetupModel().summary()

# adjust the hyper-parameters of Adam optimizer
adam_optimizer = tf.keras.optimizers.Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999, amsgrad=False)
# compile model and fit the training set
model_adam = SetupModel()
model_adam.compile(loss='binary_crossentropy', optimizer=adam_optimizer, metrics=['accuracy'])

history_adam = model_adam.fit(X_train, Y_train, 
                    epochs=epochs, batch_size=batch_size, validation_split=0.1,
                    callbacks=[EarlyStopping(monitor='val_loss', patience=3, min_delta=0.0001)])


# evaluate the result
### overall
score_train_adam = model_adam.evaluate(X_train,Y_train,verbose=0)
score_test_adam = model_adam.evaluate(X_test,Y_test,verbose=0)

# training/testing acc/loss
loss_train_adam = score_train_adam[0]
loss_test_adam = score_test_adam[0]
acc_train_adam = score_train_adam[1]
acc_test_adam = score_test_adam[1]

### epoch wise
train_acc_trend_adam = history_adam.history['accuracy']
train_loss_trend_adam = history_adam.history['loss']
val_acc_trend_adam = history_adam.history['val_accuracy']
val_loss_trend_adam = history_adam.history['val_loss']

fig, (ax1,ax2) = plt.subplots(1,2, figsize=(20,6))

#### accuarcy
result_acc_train = pd.DataFrame({
    'x':range(1,6),
    'Training': train_acc_trend_adam,
    'Validation': val_acc_trend_adam
    })

ax1.plot('x', 'Training', data=result_acc_train, marker='o', color='blue', linewidth=3,markersize=12)
ax1.plot('x', 'Validation', data=result_acc_train, marker='o', color='red', linewidth=3,markersize=12)
ax1.set_title('Prediction Accuracy for Training and Validation Set', fontsize=20)
ax1.set_xlabel('Epoch', fontsize=18)
ax1.set_ylabel('Accuracy', fontsize=16)
ax1.legend(fontsize = 10)


#### loss
result_loss_train = pd.DataFrame({
    'x':range(1,6),
    'Training': train_loss_trend_adam,
    'Validation': val_loss_trend_adam
    })

ax2.plot('x', 'Training', data=result_loss_train, marker='o', color='blue', linewidth=3,markersize=12)
ax2.plot('x', 'Validation', data=result_loss_train, marker='o', color='red', linewidth=3,markersize=12)
ax2.set_title('Binary Cross Entropy Loss for Training and Validation Set', fontsize=20)
ax2.set_xlabel('Epoch', fontsize=18)
ax2.set_ylabel('Loss', fontsize=16)
ax2.legend(fontsize = 10)


#ROC curve
from sklearn.metrics import roc_curve
y_pred_keras = model_adam.predict(X_test).ravel()
fpr_keras, tpr_keras, thresholds_keras = roc_curve(Y_test, y_pred_keras)
from sklearn.metrics import auc
auc_keras = auc(fpr_keras, tpr_keras)
plt.figure(figsize=(8,8))
plt.plot([0, 1], [0, 1], 'k--')
plt.plot(fpr_keras, tpr_keras, label='RNN Model (area = {:.3f})'.format(auc_keras))
plt.xlabel('False positive rate')
plt.ylabel('True positive rate')
plt.title('ROC curve for review classification')
plt.legend(loc='best')
plt.show()


