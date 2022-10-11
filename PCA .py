#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Importing Libraries
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import scale


# In[3]:


# Loading dataset and Renaming the columns based on their features.
columns = ['class','alcohol', 'malic_acid', 'ash', 'alcalinity_of_ash', 'magnesium',
    'total_phenols', 'flavanoids', 'nonflavanoid_phenols',
    'proanthocyanins', 'color_intensity', 'hue',
    'dilution_of_wines', 'proline']

df = pd.read_csv('wine.csv', names=columns, header=0)
df.head()


# In[4]:


# EDA & Data Preprocessing
df.info()


# In[5]:


# Checking for null values
df.isna().sum()


# In[6]:


# Checking predefined no.of cluster
df['class'].nunique()


# In[7]:


#Plot for class
df["class"].value_counts().plot.bar(color='Red')
plt.xlabel("Class")
plt.legend()


# In[8]:


# Using the standard scaler method to get the values converted into integers.
X = df.iloc[:, 1:].values
from sklearn.preprocessing import StandardScaler
X_normal = scale(X)


# In[9]:


X_normal.shape


# In[10]:


X_normal


# In[14]:


#Building PCA
''' Using Principal Component Analysis or PCA in short to reduce the dimensionality of the data in order to optimize the result 
of the clustering. '''
principalComponents = pca.fit_transform(X_normal)
pca = PCA()


# In[15]:


principalComponents


# In[16]:


# Creating a dataframe featuring the two Principal components that we acquired through PCA.
PCA_dataset = pd.DataFrame(data = principalComponents, columns = ['component1', 'component2', 'component3', 'component4', 
                                                                  'component5', 'component6','component7', 'component8', 'component9',
                                                                 'component10', 'component11', 'component12', 'component13'] )
PCA_dataset.head()


# In[28]:


# The amount of variance that each PCA explains is 
var = 'pca.explained_variance_ratio_'
var


# In[ ]:





# In[37]:


# Cumulative variance
var1 = np.cumsum(np.round(var,decimals = 4)*100)
var1


# In[35]:


'pca.components'


# In[36]:


# Variance plot for PCA components obtained 
plt.plot(var1,color="red")


# In[38]:


principal_component1 = PCA_dataset['component1']
principal_component2 = PCA_dataset['component2']
principal_component3 = PCA_dataset['component3']


# In[39]:


# Creating dataframe for further clusering algorithms
pca_df = pd.concat([principal_component1, principal_component2, principal_component3], axis = 1)
pca_df.head()


# In[40]:


# Visualizing the results of the 3D PCA.
ax = plt.figure(figsize=(10,10)).gca(projection='3d')
plt.title('3D Principal Component Analysis (PCA)')
ax.scatter(
    xs=principal_component1, 
    ys=principal_component2, 
    zs=principal_component3, 
)
ax.set_xlabel('pca-one')
ax.set_ylabel('pca-two')
ax.set_zlabel('pca-three')
plt.show()


# In[41]:


#  Normalizing Dataset
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
pca_df_normal = scaler.fit_transform(pca_df)
print(pca_df_normal)


# In[42]:


# Creating clusters
from sklearn.cluster import AgglomerativeClustering
H_clusters=AgglomerativeClustering(n_clusters=3,affinity='euclidean',linkage='ward')
H_clusters


# In[43]:


y=pd.DataFrame(H_clusters.fit_predict(pca_df_normal),columns=['clustersid_H'])
y['clustersid_H'].value_counts()


# In[45]:


# Performing K-MEANS Clustering
from sklearn.cluster import KMeans


# In[46]:


wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i,random_state=0)
    kmeans.fit(pca_df_normal)
    wcss.append(kmeans.inertia_)
    
plt.plot(range(1, 11), wcss)
plt.title('Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
plt.show()


# In[47]:


#Build Cluster algorithm
KM_clusters = KMeans(3, random_state=42)
KM_clusters.fit(pca_df_normal)


# In[48]:


y=pd.DataFrame(KM_clusters.fit_predict(pca_df_normal),columns=['clusterid_Kmeans'])
y['clusterid_Kmeans'].value_counts()


# In[49]:


# Preparing Actual Vs. Predicted Clusering Data
wine_class = df['class']
wine_class = pd.Series(wine_class)


# In[50]:


clustersid_HC = H_clusters.labels_
clustersid_HC = pd.Series(clustersid_HC)


# In[51]:


clusterid_Kmeans = KM_clusters.labels_
clusterid_Kmeans = pd.Series(clusterid_Kmeans)


# In[52]:


pred_df = pd.concat([wine_class, clustersid_HC, clusterid_Kmeans],axis = 1)
pred_df

