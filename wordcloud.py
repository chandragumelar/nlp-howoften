#!/usr/bin/env python
# coding: utf-8

# In[5]:


#get current working directory
import os 

cwd = os.getcwd()

print(cwd)


# In[6]:


#git list of file in cwd
file_list = os.listdir()

file_list


# In[7]:


import pandas as pd 

df = pd.read_csv('SDFB_people.csv')

print(df.head())


# In[10]:


#get list of columns in df
columns = df.columns.tolist()

columns


# In[13]:


from wordcloud import WordCloud
import matplotlib.pyplot as plt

#get all the text that text that is not null 
histSign = df['Historical Significance'][-pd.isnull(df['Historical Significance'])]
text = ' '.join(histSign)

#make a wordcloud
wordcloud = WordCloud(relative_scaling = 1.0).generate(text)
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.show()

