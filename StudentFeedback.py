#!/usr/bin/env python
# coding: utf-8

# In[3]:


get_ipython().system('pip install tensorflow nltk')
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM,Dense, Dropout, SpatialDropout1D
from tensorflow.keras.layers import Embedding
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
nltk.download('all')


# Data Preprocessing

# In[12]:


df = pd.read_excel('https://github.com/raneem-ewiss/StudentFeedback/blob/main/AI_Engineer_Dataset_Task_1.xlsx?raw=true')
df.dropna(subset=['ParticipantResponse'], inplace=True)
response = df[['ParticipantResponse']]
print(response)


# In[13]:


def preprocess_response(response):

    tokens = word_tokenize(response.lower())
    filtered_tokens = [token for token in tokens if token not in stopwords.words('english')]
    lemmatizer = WordNetLemmatizer()
    lemmatized_tokens = [lemmatizer.lemmatize(token) for token in filtered_tokens]
    processed_response = ' '.join(lemmatized_tokens)

    return processed_response

response = response.astype(str)
response['ParticipantResponse'] = response['ParticipantResponse'].apply(preprocess_response)
response


# Sentiment Analysis

# In[14]:


analyzer = SentimentIntensityAnalyzer()

def get_sentiment(response):
    sentiment = analyzer.polarity_scores(response)
    if sentiment['compound'] >= 0.05:
        return 'Positive'
    elif sentiment['compound'] <=-0.05:
        return 'Negative'
    else:
        return 'Neutral'

response['Sentiment'] = response['ParticipantResponse'].apply(get_sentiment)
response


# In[15]:


response_count = response.Sentiment.value_counts()
plt.bar(response_count.index, response_count.values)
plt.xlabel('Sentiment')
plt.ylabel('Count')
plt.title('Sentiment Distribution')
plt.show()

