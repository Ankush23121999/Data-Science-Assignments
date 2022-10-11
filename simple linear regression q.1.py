#!/usr/bin/env python
# coding: utf-8

# In[1]:


# import liabraries
import pandas as pd
import numpy as np
import seaborn as sns
import statsmodels.formula.api as smf


# In[3]:


# import dataset
dataset=pd.read_csv("delivery_time.csv")


# In[4]:


dataset


# In[5]:


# EDA and Data visitualization
dataset.info()


# In[11]:


sns.distplot(dataset["Delivery Time"])


# In[12]:


sns.distplot(dataset["Sorting Time"])


# In[13]:


dataset=dataset.rename({'Delivery Time':'delivery_time','Sorting Time':'Sorting_time'},axis=1)


# In[14]:


dataset


# In[15]:


# Correlation Analysis
dataset.corr()


# In[17]:


sns.regplot(x=dataset['Sorting_time'],y=dataset['delivery_time'])


# In[18]:


# model Building
model=smf.ols("delivery_time~Sorting_time",data=dataset).fit()


# In[19]:


model


# In[20]:


#model testing
model.params


# In[21]:


model.tvalues,model.pvalues


# In[26]:


model.rsquared,model.rsquared_adj


# In[27]:


# model predictions
delivery_time=(6.582734)+(1.649020)*(5)


# In[28]:


delivery_time


# In[40]:


# automatic prediction for say 3 and 5 years experience
new_data=pd.Series([3,5])


# In[37]:


new_data


# In[38]:


data_pred=pd.DataFrame(new_data,columns=['Sorting_time'])


# In[39]:


data_pred


# In[33]:


model.predict(data_pred)

