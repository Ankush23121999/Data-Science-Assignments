#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# In[2]:


salary=pd.read_csv('Salary_Data.csv')


# In[3]:


salary


# In[4]:


salary.head()


# In[5]:


salary.tail()


# In[6]:


salary.shape


# In[7]:


salary.dtypes


# In[8]:


salary.corr()


# In[9]:


sns.regplot(x=salary.YearsExperience,y=salary.Salary)


# In[11]:


import statsmodels.formula.api as smf


# In[14]:


model=smf.ols("Salary~YearsExperience",data=salary).fit()


# In[15]:


model.summary()


# In[16]:


model.params


# In[21]:


model1=smf.ols("Salary~np.log(YearsExperience)",data=salary).fit()


# In[22]:


model1.summary()


# In[23]:


model2=smf.ols("Salary~np.exp(YearsExperience)",data=salary).fit()


# In[24]:


model2.summary()


# In[25]:


model2.params


# In[26]:


pred=model.predict(salary)


# In[27]:


pred


# In[29]:


plt.scatter(x=salary.YearsExperience,y=salary.Salary,color='blue')
plt.plot(salary.YearsExperience,pred,color='black')
plt.xlabel("YearsExperience")
plt.ylabel("Salary")


# In[30]:


pred=model1.predict(salary)


# In[31]:


pred

