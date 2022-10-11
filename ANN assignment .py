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


# In[2]:


glass=pd.read_csv('glass.csv')


# In[3]:


glass.head()


# In[4]:


# split the data
x = np.array(glass.iloc[:,3:5])
y = np.array(glass['Type'])


# In[5]:


x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=1)


# In[6]:


n_neighbors = np.array(range(1,30))
param_grid = dict(n_neighbors=n_neighbors)


# In[7]:


# Grid search for finding optimal number of Neighbors
model = KNeighborsClassifier()
grid = GridSearchCV(estimator=model, param_grid=param_grid)
grid.fit(x_train, y_train)


# In[8]:


print(grid.best_score_)
print(grid.best_params_)


# In[9]:


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


# In[10]:


# Applying KNN
model = KNeighborsClassifier(n_neighbors=3)


# In[11]:


model.fit(x_train,y_train)


# In[12]:


pred=model.predict(x_test)


# In[13]:


score=accuracy_score(pred,y_test)


# In[14]:


score

