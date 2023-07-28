#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import pandas as pd
import nltk
from nltk.tokenize import word_tokenize


# In[4]:


# construct the path
cwd = os.getcwd()
file_path = os.path.join(cwd, 'data/train_author_nlp.csv')

#read the file
df = pd.read_csv(file_path)
df.head()


# In[5]:


# Group the df by author
by_author = df.groupby('author')

# Initialize ConditionalFreqDist to store word frequency by author
word_freq_by_author = nltk.probability.ConditionalFreqDist()


# In[6]:


# Iterate over each author group
for name, group in by_author:
    sentences = group['text'].str.cat(sep=' ')
    sentences = sentences.lower()
    tokens = word_tokenize(sentences)
    frequency = nltk.FreqDist(tokens)
    word_freq_by_author[name] = frequency


# In[7]:


# Calculate and display the frequency of the word 'blood' for each author

word = 'blood'

for author in word_freq_by_author.keys():
    print(f"Author: {author}")
    print(f"Frequency of {word}: {word_freq_by_author[author].freq(word)}")
    print()


# In[8]:


test_sentence = "It was a dark and stormy night."
preprocessed_test_sentence = nltk.tokenize.word_tokenize(test_sentence.lower())


# In[9]:


# Create an empty DataFrame to store author probabilities
test_probabilities = pd.DataFrame(columns=['author', 'joint_probability'])

# Iterate over each author
for author in word_freq_by_author.keys():
    joint_probability = 1.0
    
    # Calculate the joint probability for each word in the pre-processed test sentence
    for word in preprocessed_test_sentence:
        word_freq = word_freq_by_author[author].freq(word)
        smoothed_word_freq = word_freq + 0.000001
        joint_probability *= smoothed_word_freq
    
    # Create a DataFrame row with the author and its joint probability
    output = pd.DataFrame([[author, joint_probability]], columns=['author', 'joint_probability'])
    test_probabilities = pd.concat([test_probabilities, output], ignore_index=True)

# Find the author with the highest joint probability
most_likely_author = test_probabilities.loc[test_probabilities['joint_probability'].idxmax(), 'author']

# Print the joint probabilities for each author
for _, row in test_probabilities.iterrows():
    print(f"{row['author']}: {row['joint_probability']}")
    
# Print the most likely author
print(f"\nMost Likely Author: {most_likely_author}")

