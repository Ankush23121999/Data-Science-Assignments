#!/usr/bin/env python
# coding: utf-8

# In[2]:


# importing laibraries
import pandas as pd
import numpy as np
import warnings

warnings.filterwarnings("ignore")
import matplotlib.pyplot as plt
#Ols
import statsmodels.api as sm
#MSE
from sklearn.metrics import mean_squared_error
#SQRT
from math import sqrt
#ARIMA
from statsmodels.tsa.arima_model import ARIMA

#inline Visualization
 # %matplotlib inli


# In[3]:


# Loading dataset
cocacola = pd.read_excel('CocaCola_Sales_Rawdata.xlsx')


# In[4]:


#copy
df = cocacola.copy()


# In[5]:


#head
df.head()


# In[6]:


#index
df = df.set_index('Quarter')


# In[7]:


df.head()


# In[8]:


#line plot
df.plot()
plt.show()


# In[9]:


#histogram
df.hist()


# In[10]:


#kde
df.plot(kind='kde')


# In[11]:


# load data
train = pd.read_excel('CocaCola_Sales_Rawdata.xlsx', header=0, index_col=0, parse_dates=True)


# In[12]:


# prepare data
X = train.values
X = X.astype('float32')
train_size = int(len(X) * 0.50)
train, test = X[0:train_size], X[train_size:]


# In[13]:


train


# In[14]:


history = [x for x in train]
predictions = list()
for i in range(len(test)):
    yhat = history[-1]
    predictions.append(yhat)
# observation
    obs = test[i]
    history.append(obs)
    print('>Predicted=%.3f, Expected=%.3f' % (yhat, obs))
# report performance
rmse = sqrt(mean_squared_error(test, predictions))
print('RMSE: %.3f' % rmse)


# In[15]:


# evaluate an ARIMA model for a given order (p,d,q) and return RMSE
def evaluate_arima_model(X, arima_order):
# prepare training dataset
    X = X.astype('float32')
    train_size = int(len(X) * 0.50)
    train, test = X[0:train_size], X[train_size:]
    history = [x for x in train]
# make predictions
    predictions = list()
    for t in range(len(test)):
        model = ARIMA(history, order=arima_order)
# model_fit = model.fit(disp=0)
        model_fit = model.fit(disp=0)
        yhat = model_fit.forecast()[0]
        predictions.append(yhat)
        history.append(test[t])
# calculate out of sample error
    rmse = sqrt(mean_squared_error(test, predictions))
    return rmse


# In[16]:


# evaluate combinations of p, d and q values for an ARIMA model
def evaluate_models(dataset, p_values, d_values, q_values):
    dataset = dataset.astype('float32')
    best_score, best_cfg = float('inf'), None
    for p in p_values:
        for d in d_values:
            for q in q_values:
                order = (p,d,q)
                try:
                    rmse = evaluate_arima_model(train, order)
                    if rmse < best_score:
                        best_score, best_cfg = rmse, order
                    print('ARIMA%s RMSE=%.3f' % (order,rmse))
                except:
                    continue
    print('Best ARIMA%s RMSE=%.3f' % (best_cfg, best_score))


# In[17]:


train = pd.read_excel('CocaCola_Sales_Rawdata.xlsx', header=0, index_col=0, parse_dates=True)
X = train.values
X = X.astype('float32')


# In[18]:


# fit model
model = ARIMA(X, order=(3,1,0))
model_fit = model.fit()
forecast=model_fit.forecast(steps=10)[0]
model_fit.plot_predict(1, 79)

