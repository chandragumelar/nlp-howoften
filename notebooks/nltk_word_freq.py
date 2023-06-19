#!/usr/bin/env python
# coding: utf-8

# In[6]:


import os
import pandas as pd
import nltk
from nltk.tokenize import word_tokenize


# In[5]:


# construct the path
file_path = os.path.join(cwd, 'data/train_author_nlp.csv')

#read the file
df = pd.read_csv(file_path)
df.head()


# In[9]:


# Group the df by author
byAuthor = df.groupby('author')

# Initialize ConditionalFreqDist to store word frequency by author
wordFreqByAuthor = nltk.probability.ConditionalFreqDist()


# In[10]:


# Iterate over each author group
for name, group in byAuthor:
    sentences = group['text'].str.cat(sep=' ')
    sentences = sentences.lower()
    tokens = word_tokenize(sentences)
    frequency = nltk.FreqDist(tokens)
    wordFreqByAuthor[name] = frequency


# In[16]:


# Calculate and display the frequency of the word 'blood' for each author

word = 'blood'

for author in wordFreqByAuthor.keys():
    print(f"Author: {author}")
    print(f"Frequency of {word}: {wordFreqByAuthor[author].freq(word)}")
    print()


# In[21]:


testSentence = "It was a dark and stormy night."
preProcessedTestSentence = nltk.tokenize.word_tokenize(testSentence.lower())


# In[27]:


testProbabilities = pd.DataFrame(columns=['author', 'jointProbability'])

for author in wordFreqByAuthor.keys():
    jointProbability = 1.0
    
    for word in preProcessedTestSentence:
        wordFreq = wordFreqByAuthor[author].freq(word)
        smoothedWordFreq = wordFreq + 0.000001
        jointProbability *= smoothedWordFreq
    
    output = pd.DataFrame([[author, jointProbability]], columns=['author', 'jointProbability'])
    testProbabilities = pd.concat([testProbabilities, output], ignore_index=True)

mostLikelyAuthor = testProbabilities.loc[testProbabilities['jointProbability'].idxmax(), 'author']

for _, row in testProbabilities.iterrows():
    print(f"{row['author']}: {row['jointProbability']}")

print(f"\nMost Likely Author: {mostLikelyAuthor}")

