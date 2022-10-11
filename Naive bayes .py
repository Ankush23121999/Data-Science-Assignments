#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

#Naive Bayes
from sklearn.naive_bayes import MultinomialNB as MB
from sklearn.naive_bayes import GaussianNB as GB

get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


train = pd.read_csv('SalaryData_Train.csv')
test = pd.read_csv('SalaryData_Test.csv')


# In[3]:


#copying the data
df1 = train.copy()
df2 = test.copy()


# In[4]:


#Head
df1.head()


# In[5]:


df2.head()


# In[6]:


df1.isnull().sum()


# In[7]:


df2.isnull().sum()


# In[8]:


df1.info()


# In[9]:


df2.info()


# In[10]:


#workclass
plt.figure(figsize=(12,5))
df1.workclass.value_counts().plot.bar(color='yellow');


# In[11]:


#maritial status
plt.figure(figsize=(12,5))
df1.maritalstatus.value_counts().plot.bar(color='green');


# In[12]:


#occupation
plt.figure(figsize=(12,5))
df1.occupation.value_counts().plot.bar(color='orange');


# In[13]:


#relationship
plt.figure(figsize=(12,5))
df1.relationship.value_counts().plot.bar(color='pink');


# In[14]:


#race
plt.figure(figsize=(12,5))
df1.race.value_counts().plot.bar(color='grey');


# In[15]:


#sex
plt.figure(figsize=(12,5))
df1.sex.value_counts().plot.bar(color='blue');


# In[16]:


#salary
plt.figure(figsize=(12,5))
df1.Salary.value_counts().plot.bar(color='purple');


# In[17]:


# one hot encoding
tr1 = df1.iloc[:,0:13]

tr1 = pd.get_dummies(tr1)
tr1


# In[18]:


te1 = test.iloc[:,0:13]

te1 = pd.get_dummies(te1)
te1


# In[19]:


#train data concat
finaltrain = pd.concat([tr1, df1['Salary']],axis=1)
finaltrain


# In[20]:


#test data concat
finaltest = pd.concat([te1, df2['Salary']],axis=1)
finaltest


# In[21]:


# Dividing Finaltrain and Finaltest Data
# Finaltrain data
X = finaltrain.values[:,0:102]
Y = finaltrain.values[:,102]

#Finaltest data
x = finaltest.values[:,0:102]
y = finaltest.values[:,102]


# In[22]:


# Naive Bayes Model
classifier_mb = MB()
classifier_mb.fit(X,Y)
train_pred_m = classifier_mb.predict(X)
accuracy_train_m = np.mean(train_pred_m==Y)


# In[23]:


test_pred_m = classifier_mb.predict(x)
accuracy_test_m = np.mean(test_pred_m==y)


# In[24]:


print('Training accuracy is:',accuracy_train_m,'\n','Testing accuracy is:',accuracy_test_m)


# In[25]:


# Gaussian Naive Bayes
classifier_gb = GB()
classifier_gb.fit(X,Y) 
# we need to convert tfidf into array format which is compatible for gaussian naive bayes
train_pred_g = classifier_gb.predict(X)
accuracy_train_g = np.mean(train_pred_g==Y)


# In[26]:


test_pred_g = classifier_gb.predict(x)
accuracy_test_g = np.mean(test_pred_g==y)


# In[27]:


print('Training accuracy is:',accuracy_train_g,'\n','Testing accuracy is:',accuracy_test_g)

