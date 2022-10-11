#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score


# In[4]:


zoo=pd.read_csv('zoo.csv')


# In[5]:


zoo.head()


# In[6]:


# split the data
x = zoo.iloc[:,1:17]
y = zoo.iloc[:,17]


# In[7]:


x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=1)


# In[8]:


# Grid search for finding optimal number of Neighbors
n_neighbors = np.array(range(1,30))
param_grid = dict(n_neighbors=n_neighbors)


# In[9]:


model = KNeighborsClassifier()
grid = GridSearchCV(estimator=model, param_grid=param_grid)
grid.fit(x_train, y_train)


# In[10]:


print(grid.best_score_)
print(grid.best_params_)


# In[11]:


k_range = range(1, 31)
k_scores = []
for k in k_range:
    knn = KNeighborsClassifier(n_neighbors=k)
    train_scores = cross_val_score(knn, x_train, y_train, cv=5)
    k_scores.append(train_scores.mean())
plt.plot(k_range, k_scores)
plt.xlabel('Value of K for KNN')
plt.ylabel('Cross-Validated Accuracy')
plt.show()


# In[12]:


# Applying KNN
model = KNeighborsClassifier(n_neighbors=1)


# In[13]:


model.fit(x_train,y_train)


# In[14]:


pred=model.predict(x_test)


# In[15]:


score=accuracy_score(pred,y_test)


# In[16]:


score

