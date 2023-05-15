#!/usr/bin/env python
# coding: utf-8

# In[11]:


import pandas as pd

df = pd.read_csv('Documents/py-playground/train_author_nlp.csv')

df.head(5)


# In[12]:


import nltk

# split data by the author
byAuthor = df.groupby('author')

# word frequency by author in dict
wordFreqByAuthor = nltk.probability.ConditionalFreqDist()

#for each author
for name, group in byAuthor:
    #get all the sentences they wrote and collapse it into single string
    sentences = group['text'].str.cat(sep=' ')
    #get all the sentences in lower case
    sentences = sentences.lower()
    #split text into individual tokens
    tokens = nltk.tokenize.word_tokenize(sentences)
    #calculate the frequency of each tokens
    frequency = nltk.FreqDist(tokens)
    #add the frequencies of each author into dictionary
    wordFreqByAuthor[name] = (frequency)


# In[14]:


#see how often each author says "blood"
for i in wordFreqByAuthor.keys():
    print('blood :' +i)
    print(wordFreqByAuthor[i].freq('blood'))


# In[31]:


# first, let's start with a test sentence
testSentence = "It was a dark and stormy night."

# and then lowercase & tokenize our test sentence
preProcessedTestSentence = nltk.tokenize.word_tokenize(testSentence.lower())

# create an empy dataframe to put our output in
testProbailities = pd.DataFrame(columns = ['author','word','probability'])

# For each author...
for i in wordFreqByAuthor.keys():
    # for each word in our test sentence...
    for j  in preProcessedTestSentence:
        # find out how frequently the author used that word
        wordFreq = wordFreqByAuthor[i].freq(j)
        # and add a very small amount to every prob. so none of them are 0
        smoothedWordFreq = wordFreq + 0.000001
        # add the author, word and smoothed freq. to our dataframe
        output = pd.DataFrame([[i, j, smoothedWordFreq]], columns = ['author','word','probability'])
        testProbailities = pd.concat([testProbailities, output], ignore_index=True)
        
# empty df for the probability that each author wrote the sentence
testProbabilitiesByAuthor = pd.DataFrame(columns = ['author','jointProbability'])

# lets group the df with frequency by author
for i in wordFreqByAuthor.keys():
    # get the joint probability that each author wrote each word
    oneAuthor = testProbabilities.query('author == "' + i + '"')
    jointProbability = oneAuthor.product(numeric_only = True)[0]
    
    # and add that to our dataframe
    output = pd.DataFrame([[i, jointProbability]], columns = ['author','jointProbability'])
    testProbabilitiesByAuthor = pd.concat([testProbabilitiesByAuthor, output], ignore_index=True)

# and our winner is...
testProbabilitiesByAuthor.loc[testProbabilitiesByAuthor['jointProbability'].idxmax(),'author']

