# Word Embeddings and Recurrent Neural Networks For Disaster Tweet Classification

Boyang Wei, Lechuan Qiu

## Objective 

This project focuses on using different word embedding and recurrent neural networks to classify [Tweet disaster data from Kaggle Competition](https://www.kaggle.com/c/nlp-getting-started/data). We are exploring how TF-IDF, Word2Vec, LSTM and BERT performed on binary classification task, specifically to this sample dataset.

## Data

Tweet disaster data from Kaggle Competition are used for the entire pipeline from data cleaning, modeling to evaluation. The data consisted of a training and testing file in form of common-separated values. Training and testing files contain 3243 and 7503 unique rows and 4 features. Cleaned data could be found in data directory as cleaned.csv.

## Background

Traditional natural language processing for classification problems rely on proper cleaning, careful choice of word embedding and machine learning models. Texts data are treated differently from numerical data which could be statistically normalized. For tweets, texts needed to be ‘normalized’ as proper English and Non-Roman characters sometimes need to be removed. Stop words, symbols and punctuations are usually removed for training. After cleaning the data, this paper examines different combinations of word embeddings and machine learning models. We also proposed some potential explanations on the results from best to worst combinations which provide valuable insights for further Tweet analysis.

## Methodology 

We first explore and clean the original tweet dataset based on results from EDA, which could be found in [EDA and Cleaning.py](EDA%20and%20Cleaning.py)
We are exploring 2 combinations of **word embedding + logistic regression** and 2 types of **recurrent neural networks**. The details and codes could be found in [Recurrent Neural Networks - LSTM.py](Recurrent%20Neural%20Networks%20-%20LSTM.py), [Recurrent Neural Networks - BERT.py](Recurrent%20Neural%20Networks%20-%20BERT.py) and [TFIDF+ W2V Logistic Regression.ipynb](TFIDF+%20W2V%20Logistic%20Regression.ipynb)  

For our neural embedding, we choose to use the w2v pretrain model from Google, which is trained with the Google News dataset (about 100 billion words). And the final model contains 300 dimensional vectors for 3 million words and phrases. So it should very much cover all the words in our disaster twitter dataset. In order to run the code, the pretrain model is required. Download link: https://drive.google.com/file/d/0B7XkCwpI5KDYNlNUTTlSS21pQmM/edit?source=post_page (This file is 1.65GB, so we were not able to push to github)

## Results

Based on prediction results and training, validation loss and precision, BERT performed the best among all four models and TF-IDF tends to perform the worst as baseline model. The details of metrics could be found in [Result Paper.docx](Result%20Paper.docx)

## Discussion

For TF-IDF, we believe that since tweets are written by multiple users, the common words in each tweet could be a good reference and implication of the class, the overall sample size is still low compared to the total number of tweets. Therefore, TF-IDF potentially captured the most common words and assign the weights but neglect the large portion of terms that also have important inference. 
Word2Vec tends to perform better than TF-IDF, given that tweets could be diverse in terms of word choice. Clustering different terms of the similar meaning enable the computer to train relatively simple model. Given that both deep neural networks show overfitting with even regular number of epochs, Word2Vec is a good choice in our sample dataset.
For LSTM and BERT, we believe that this task for classifying disaster and non-disaster tweet is relatively simple. As training log shows the model gets overfit to epoch 3 and 4, the models are clearly too flexible for the tweet. Therefore, deep neural networks, even though BERT scores the highest in term of prediction accuracy, are not necessary and appropriate to use for this task. Simple model such as Word2Vec and logistic regression also scores 0.81 accuracy rate and are easier to interpret. We will continue to work on different word embeddings and deep learning models to better decide if we really need to have complex models to better capture details that traditional models could not learn and interpret. 
