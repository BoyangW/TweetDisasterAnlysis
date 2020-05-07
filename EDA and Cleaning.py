# -*- coding: utf-8 -*-
"""
This file consists of code and explanations on exploratory data analysis and 
cleaning of Disaster Tweets from Kaggle Competition
@author: Boyang Wei
"""

import string
from collections import defaultdict
from wordcloud import STOPWORDS
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import re

"""
Exploratory Data Analysis
The data consists of 2 files: train.csv and test.csv
The train file contains five features: ID, Keyword, Location, Text and Target 
(Label 1 for disaster and 0 for non-disaster)
"""

df_train = pd.read_csv('data/train.csv', dtype={'id': np.int16, 'target': np.int8})
df_test = pd.read_csv('data/test.csv', dtype={'id': np.int16})
print('Training Set Shape = {}'.format(df_train.shape))
print('Test Set Shape = {}'.format(df_test.shape))

#Sample raw data in training set
df_train.head(5)

def missingValues():
    missing_cols = ['keyword', 'location']

    fig, axes = plt.subplots(ncols=2, figsize=(17, 4), dpi=100)
    
    sns.barplot(x=df_train[missing_cols].isnull().sum().index, y=df_train[missing_cols].isnull().sum().values, ax=axes[0])
    sns.barplot(x=df_test[missing_cols].isnull().sum().index, y=df_test[missing_cols].isnull().sum().values, ax=axes[1])
    
    axes[0].set_ylabel('Missing Value Count', size=15, labelpad=20)
    axes[0].tick_params(axis='x', labelsize=15)
    axes[0].tick_params(axis='y', labelsize=15)
    axes[1].tick_params(axis='x', labelsize=15)
    axes[1].tick_params(axis='y', labelsize=15)
    
    axes[0].set_title('Training Set', fontsize=13)
    axes[1].set_title('Test Set', fontsize=13)
    
    plt.show()
    
    for df in [df_train, df_test]:
        for col in ['keyword', 'location']:
            df[col] = df[col].fillna(f'no_{col}')
            
def classDitribution():
    Real_len = df_train[df_train['target'] == 1].shape[0]
    Not_len = df_train[df_train['target'] == 0].shape[0]
    # bar plot of the 3 classes
    plt.rcParams['figure.figsize'] = (7, 5)
    plt.bar(10,Real_len,3, label="Real", color='blue')
    plt.bar(15,Not_len,3, label="Not", color='red')
    plt.legend()
    plt.ylabel('Number of examples')
    plt.title('Propertion of examples')
    plt.show()
    
    print("Number of real disaster tweets:", Real_len)
    print("Number of non-disaster tweets:", Not_len)
    
    
#get length of text for Text column
def length(text):    
    return len(text)

#get number of characters of texts in both classes (distribution)
def getCharac():
    df_train['length'] = df_train['text'].apply(length)
    
    fig,(ax1,ax2)=plt.subplots(1,2,figsize=(10,5))
    tweet_len=df_train[df_train['target']==1]['text'].str.len()
    ax1.hist(tweet_len,color='blue')
    ax1.set_title('Disaster tweets')
    tweet_len=df_train[df_train['target']==0]['text'].str.len()
    ax2.hist(tweet_len,color='red')
    ax2.set_title('Non-disaster tweets')
    fig.suptitle('Characters in tweets')
    plt.show()

#get top 30 frequent unigrams, bigrams and trigrams for disaster and non-disaster tweets
def generate_ngrams(text, n_gram=1):
    token = [token for token in text.lower().split(' ') if token != '' if token not in STOPWORDS]
    ngrams = zip(*[token[i:] for i in range(n_gram)])
    return [' '.join(ngram) for ngram in ngrams]

DISASTER_TWEETS = df_train['target'] == 1
N = 30

#unigrams
def Unigram():
    disaster_unigrams = defaultdict(int)
    nondisaster_unigrams = defaultdict(int)
    
    for tweet in df_train[DISASTER_TWEETS]['text']:
        for word in generate_ngrams(tweet):
            disaster_unigrams[word] += 1
            
    for tweet in df_train[~DISASTER_TWEETS]['text']:
        for word in generate_ngrams(tweet):
            nondisaster_unigrams[word] += 1
            
    df_disaster_unigrams = pd.DataFrame(sorted(disaster_unigrams.items(), key=lambda x: x[1])[::-1])
    df_nondisaster_unigrams = pd.DataFrame(sorted(nondisaster_unigrams.items(), key=lambda x: x[1])[::-1])
    
    fig, axes = plt.subplots(ncols=2, figsize=(18, 50), dpi=100)
    plt.tight_layout()
    
    sns.barplot(y=df_disaster_unigrams[0].values[:N], x=df_disaster_unigrams[1].values[:N], ax=axes[0], color='red')
    sns.barplot(y=df_nondisaster_unigrams[0].values[:N], x=df_nondisaster_unigrams[1].values[:N], ax=axes[1], color='green')
    
    for i in range(2):
        axes[i].spines['right'].set_visible(False)
        axes[i].set_xlabel('')
        axes[i].set_ylabel('')
        axes[i].tick_params(axis='x', labelsize=13)
        axes[i].tick_params(axis='y', labelsize=13)
    
    axes[0].set_title(f'Top {N} most common unigrams in Disaster Tweets', fontsize=15)
    axes[1].set_title(f'Top {N} most common unigrams in Non-disaster Tweets', fontsize=15)
    
    plt.show()
    
#bigrams
def Bigram():
    disaster_bigrams = defaultdict(int)
    nondisaster_bigrams = defaultdict(int)
    
    for tweet in df_train[DISASTER_TWEETS]['text']:
        for word in generate_ngrams(tweet, n_gram=2):
            disaster_bigrams[word] += 1
            
    for tweet in df_train[~DISASTER_TWEETS]['text']:
        for word in generate_ngrams(tweet, n_gram=2):
            nondisaster_bigrams[word] += 1
            
    df_disaster_bigrams = pd.DataFrame(sorted(disaster_bigrams.items(), key=lambda x: x[1])[::-1])
    df_nondisaster_bigrams = pd.DataFrame(sorted(nondisaster_bigrams.items(), key=lambda x: x[1])[::-1])
    
    fig, axes = plt.subplots(ncols=2, figsize=(18, 50), dpi=100)
    plt.tight_layout()
    
    sns.barplot(y=df_disaster_bigrams[0].values[:N], x=df_disaster_bigrams[1].values[:N], ax=axes[0], color='red')
    sns.barplot(y=df_nondisaster_bigrams[0].values[:N], x=df_nondisaster_bigrams[1].values[:N], ax=axes[1], color='green')
    
    for i in range(2):
        axes[i].spines['right'].set_visible(False)
        axes[i].set_xlabel('')
        axes[i].set_ylabel('')
        axes[i].tick_params(axis='x', labelsize=13)
        axes[i].tick_params(axis='y', labelsize=13)
    
    axes[0].set_title(f'Top {N} most common bigrams in Disaster Tweets', fontsize=15)
    axes[1].set_title(f'Top {N} most common bigrams in Non-disaster Tweets', fontsize=15)
    
    plt.show()
    
#trigrams
def Trigram():
    disaster_trigrams = defaultdict(int)
    nondisaster_trigrams = defaultdict(int)
    
    for tweet in df_train[DISASTER_TWEETS]['text']:
        for word in generate_ngrams(tweet, n_gram=3):
            disaster_trigrams[word] += 1
            
    for tweet in df_train[~DISASTER_TWEETS]['text']:
        for word in generate_ngrams(tweet, n_gram=3):
            nondisaster_trigrams[word] += 1
            
    df_disaster_trigrams = pd.DataFrame(sorted(disaster_trigrams.items(), key=lambda x: x[1])[::-1])
    df_nondisaster_trigrams = pd.DataFrame(sorted(nondisaster_trigrams.items(), key=lambda x: x[1])[::-1])
    
    fig, axes = plt.subplots(ncols=2, figsize=(20, 50), dpi=100)
    
    sns.barplot(y=df_disaster_trigrams[0].values[:N], x=df_disaster_trigrams[1].values[:N], ax=axes[0], color='red')
    sns.barplot(y=df_nondisaster_trigrams[0].values[:N], x=df_nondisaster_trigrams[1].values[:N], ax=axes[1], color='green')
    
    for i in range(2):
        axes[i].spines['right'].set_visible(False)
        axes[i].set_xlabel('')
        axes[i].set_ylabel('')
        axes[i].tick_params(axis='x', labelsize=13)
        axes[i].tick_params(axis='y', labelsize=11)
    
    axes[0].set_title(f'Top {N} most common trigrams in Disaster Tweets', fontsize=15)
    axes[1].set_title(f'Top {N} most common trigrams in Non-disaster Tweets', fontsize=15)
    
    plt.show()

#Run Functions Above:
missingValues()
classDitribution()
getCharac()
Unigram()
Bigram()
Trigram()
##############################################################################
#Data Cleaning
#From the EDA part above, we can see most of the key words and locations are missing from training set.
#Therefore, only texts are used for further classification task. The data cleaning part will focus on 
#using regular expression to remove irrelevant information such as links, HTML tags, symbols, punctuations, 
#stop words


#Remove Links
def remove_URL(text):
    url = re.compile(r'https?://\S+|www\.\S+')
    return url.sub(r'',text)

df_train['text']=df_train['text'].apply(lambda x : remove_URL(x))


#Remove HTML
def remove_html(text):
    html=re.compile(r'<.*?>')
    return html.sub(r'',text)
df_train['text']=df_train['text'].apply(lambda x : remove_html(x))


#Remove Emojis
def remove_emoji(text):
    emoji_pattern = re.compile("["
                           u"\U0001F600-\U0001F64F"  # emoticons
                           u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                           u"\U0001F680-\U0001F6FF"  # transport & map symbols
                           u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                           u"\U00002702-\U000027B0"
                           u"\U000024C2-\U0001F251"
                           "]+", flags=re.UNICODE)
    return emoji_pattern.sub(r'', text)
df_train['text']=df_train['text'].apply(lambda x : remove_emoji(x))


#Remove Punctuations
def remove_punct(text):
    table=str.maketrans('','',string.punctuation)
    return text.translate(table)

df_train['text']=df_train['text'].apply(lambda x : remove_punct(x))


#Save the result
df_save = df_train[['id','text', 'target']]
df_save.to_csv('data/clean.csv', index=False)