#!/usr/bin/env python
# coding: utf-8

# In[2]:


# importing laibraries
import pandas as pd
import numpy as np
from sklearn.metrics import pairwise_distances
from scipy.spatial.distance import cosine,correlation


# In[3]:


# Import Dataset
books=pd.read_csv('book.csv',encoding='Latin1')
books


# In[4]:


books2=books.iloc[:,1:]
books2


# In[5]:


# Sort by User IDs
books2.sort_values(['User.ID'])


# In[6]:


# number of unique users in the dataset
len(books2['User.ID'].unique())


# In[7]:


# number of unique books in the dataset
len(books2['Book.Title'].unique())


# In[8]:


# converting long data into wide data using pivot table
books3=books2.pivot_table(index='User.ID',columns='Book.Title',values='Book.Rating').reset_index(drop=True)
books3


# In[9]:


# Replacing the index values by unique user Ids
books3.index=books2['User.ID'].unique()
books3


# In[10]:


# Impute those NaNs with 0 values
books3.fillna(0,inplace=True)
books3


# In[11]:


# Calculating Cosine Similarity between Users on array data
user_sim=1-pairwise_distances(books3.values,metric='cosine')
user_sim


# In[12]:


# Store the results in a dataframe format
user_sim2=pd.DataFrame(user_sim)
user_sim2


# In[13]:


# Set the index and column names to user ids 
user_sim2.index=books2['User.ID'].unique()
user_sim2.columns=books2['User.ID'].unique()
user_sim2


# In[14]:


# Nullifying diagonal values
np.fill_diagonal(user_sim,0)
user_sim2


# In[15]:


# Most Similar Users
user_sim2.idxmax(axis=1)


# In[16]:


# extract the books which userId 162107 & 276726 have watched
books2[(books2['User.ID']==162107) | (books2['User.ID']==276726)]


# In[17]:


# extract the books which userId 276729 & 276726 have watched
books2[(books2['User.ID']==276729) | (books2['User.ID']==276726)]


# In[18]:


user_1=books2[(books2['User.ID']==276729)]
user_2=books2[(books2['User.ID']==276726)]


# In[19]:


user_1['Book.Title']


# In[20]:


user_1['Book.Title']


# In[21]:


pd.merge(user_1,user_2,on='Book.Title',how='outer')

