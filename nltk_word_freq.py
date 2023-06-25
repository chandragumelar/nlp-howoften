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


# Create an empty DataFrame to store author probabilities
testProbabilities = pd.DataFrame(columns=['author', 'jointProbability'])

# Iterate over each author
for author in wordFreqByAuthor.keys():
    jointProbability = 1.0
    
    # Calculate the joint probability for each word in the pre-processed test sentence
    for word in preProcessedTestSentence:
        wordFreq = wordFreqByAuthor[author].freq(word)
        smoothedWordFreq = wordFreq + 0.000001
        jointProbability *= smoothedWordFreq
    
    # Create a DataFrame row with the author and its joint probability
    output = pd.DataFrame([[author, jointProbability]], columns=['author', 'jointProbability'])
    testProbabilities = pd.concat([testProbabilities, output], ignore_index=True)

# Find the author with the highest joint probability
mostLikelyAuthor = testProbabilities.loc[testProbabilities['jointProbability'].idxmax(), 'author']

# Print the joint probabilities for each author
for _, row in testProbabilities.iterrows():
    print(f"{row['author']}: {row['jointProbability']}")
    
# Print the most likely author
print(f"\nMost Likely Author: {mostLikelyAuthor}")

