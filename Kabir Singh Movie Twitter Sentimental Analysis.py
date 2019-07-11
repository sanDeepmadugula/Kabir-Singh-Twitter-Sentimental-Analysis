#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd

import os
os.chdir('C:\\Analytics\\MachineLearning\\Kabir Singh Twitter Sentiment Analysis')


# In[2]:


data = pd.read_csv('twitter_kabir_singh_bollywood_movie.csv', delimiter=',')

data.head()


# In[3]:


data['author'].value_counts()


# 8685 different users have twitted on the movie.

# In[4]:


text = data['text_raw']
import nltk.classify.util
from nltk.classify import NaiveBayesClassifier
from nltk.corpus import names

def word_feats(words):
    return dict([word,True] for word in words)


positive_vocab = ['awesome', 'outstanding', 'fantastic', 'terrific', 'good', 'nice','great']

negative_vocab = ['bad', 'terrible', 'useless', 'hate']

neutral_vocab = ['movie', 'the', 'sound','was', 'is', 'actors', 'did', 'know', 'words','not']

positive_features = [(word_feats(pos), 'pos') for pos in positive_vocab]

negative_features = [(word_feats(neg), 'neg') for neg in negative_vocab]

neutral_features = [(word_feats(neu), 'neu') for neu in neutral_vocab]

train_set = positive_features + negative_features + neutral_features

classifier = NaiveBayesClassifier.train(train_set)


# In[5]:


total_text = pd.DataFrame()
total_text['text'] = text
total_text['sentiment'] = 'neu'

for i in range(total_text.shape[0]):
    words = total_text['text'][i].split(' ')
    pos = 0
    neg = 0
    for word in words:
        classResult = classifier.classify(word_feats(word))
        
        if classResult == 'neg':
            neg = neg+1
        if classResult == 'pos':
            pos = pos+1
            
    if pos > neg:
        total_text['sentiment'][i] = 'pos'
    if neg > pos:
        total_text['sentiment'][i] = 'neg'
        
        
print(total_text.head())


# In[6]:


total_text['sentiment'].value_counts()


# Most of the tweets are positive for the movie.

# In[7]:


total_text['favorite_count'] = data['favorite_count']
total_text['reply_count'] = data['reply_count']

total_text.sort_values(by=['favorite_count'],ascending=False).head(6)


# In[8]:


total_text.loc[total_text['sentiment'] == 'neg'].sort_values(by=['favorite_count'],ascending=False).head(6)


# In[9]:


total_text.sort_values(by=['reply_count'],ascending=False).head(6)


# In[11]:



total_text.loc[total_text['sentiment']=='neg'].sort_values(by=['reply_count'], ascending = False).head(10)


# The conclusion is that, positive tweets include significantly more
# favorites than negative ones. The most popular tweet has 7822 favourites
# and 175 replies while the most important negative tweet has only 60
# favourites and only 3 replies.
# 
# 

# In[ ]:




