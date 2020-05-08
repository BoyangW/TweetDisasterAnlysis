# load library
import logging
import pandas as pd
import numpy as np
from numpy import random
import gensim
import nltk
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
from nltk.corpus import stopwords

from sklearn.feature_extraction.text import TfidfVectorizer
from gensim.models import Word2Vec

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

def word_averaging(wv, words):
    all_words, mean = set(), []

    for word in words:
        if isinstance(word, np.ndarray):
            mean.append(word)
        elif word in wv.vocab:
            mean.append(wv.syn0norm[wv.vocab[word].index])
            all_words.add(wv.vocab[word].index)

    if not mean:
        logging.warning("cannot compute similarity with no input %s", words)
        # FIXME: remove these examples in pre-processing
        return np.zeros(wv.vector_size, )

    mean = gensim.matutils.unitvec(np.array(mean).mean(axis=0)).astype(np.float32)
    return mean


def word_averaging_list(wv, text_list):
    return np.vstack([word_averaging(wv, post) for post in text_list])


# tokenize
def w2v_tokenize_text(text):
    tokens = []
    for sent in nltk.sent_tokenize(text, language='english'):
        for word in nltk.word_tokenize(sent, language='english'):
            if len(word) < 2:
                continue
            tokens.append(word)
    return tokens




# load data
df_clean = pd.read_csv('data/clean.csv', dtype={'id': np.int16, 'target': np.int8})



#### Tfidf
# convert to tfidf
tfidfconverter = TfidfVectorizer(max_features=1500, min_df=5, max_df=0.7, stop_words=stopwords.words('english'))
X = tfidfconverter.fit_transform(df_clean['text']).toarray()
y = df_clean['target']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

# run logistic
logreg = LogisticRegression(n_jobs=1, C=1e5, max_iter=5000)
logreg = logreg.fit(X_train, y_train)
y_pred = logreg.predict(X_test)
print('Following are results for logistic regression with tfidf')
print('accuracy %s' % accuracy_score(y_pred, y_test))
print(classification_report(y_test, y_pred))


### w2v
# convert to w2v
# download the pretrain model: https://drive.google.com/file/d/0B7XkCwpI5KDYNlNUTTlSS21pQmM/edit?source=post_page
wv = gensim.models.KeyedVectors.load_word2vec_format("GoogleNews-vectors-negative300.bin.gz", binary=True)
wv.init_sims(replace=True)

nltk.download('all')


# splict train test set
train, test = train_test_split(df_clean, test_size=0.3, random_state = 42)

# average
test_tokenized = test.apply(lambda r: w2v_tokenize_text(r['text']), axis=1).values
train_tokenized = train.apply(lambda r: w2v_tokenize_text(r['text']), axis=1).values

X_train_word_average = word_averaging_list(wv,train_tokenized)
X_test_word_average = word_averaging_list(wv,test_tokenized)



logreg = LogisticRegression(n_jobs=1, C=1e5, max_iter = 5000)
logreg = logreg.fit(X_train_word_average, train['target'])
y_pred = logreg.predict(X_test_word_average)
print('Following are results for logistic regression with w2v')
print('accuracy %s' % accuracy_score(y_pred, test.target))
print(classification_report(test.target, y_pred))