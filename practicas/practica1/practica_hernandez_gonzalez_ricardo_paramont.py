#!/usr/bin/env python
# coding: utf-8

# # Práctica 
# 
# #### Hernández González Ricardo Paramont
# 
# **********************

# ### Dependencias 

# In[1]:


import numpy as np
import pandas as pd 

from sklearn.preprocessing import MinMaxScaler,StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import MDS,TSNE
from sklearn.cluster import AgglomerativeClustering,KMeans
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score

from scipy.stats import chisquare
from scipy.stats import kruskal
from statsmodels.stats.multicomp import MultiComparison

from itertools import chain
from functools import reduce

import matplotlib.pyplot as plt
import seaborn as sns
import cufflinks as cf
import plotly.express as px
import plotly.graph_objects as go

cf.go_offline()
pd.set_option('display.max_columns',None)


# ### Importación de datos

# In[2]:


df = pd.read_csv('atm.csv')
df.head()


# In[3]:


len(df)


# ### Análisis exploratorio 

# #### Ausentes

# In[4]:


miss = 1-df.describe().T[['count']]/len(df)
miss.sort_values(by='count',ascending=False)


# #### Distribuciones

# In[5]:


df['ts_transaction_date'].hist()


# ### Ingeniería de variables

# In[6]:


df.dtypes


# In[7]:


df['ts_transaction_date'] = pd.to_datetime(df['ts_transaction_date'])


# In[8]:


df.dtypes


# In[40]:


# confirmamos que solo hay registros del año 2017
np.unique(df['ts_transaction_date'].dt.year)


# #### Obteniendo semestre

# In[9]:


df['semestre']=(df['ts_transaction_date'].dt.month>6).astype(int)+1


# #### Obteniendo cuatrimestre

# In[10]:


df['cuatrimestre']=(df['ts_transaction_date'].dt.month/4).apply(np.ceil)


# #### Obteniendo bimestre

# In[11]:


df['bimestre']=(df['ts_transaction_date'].dt.month/2).apply(np.ceil)


# visualizando

# In[12]:


varc = ['semestre','cuatrimestre','bimestre']
df[varc].hist()


# In[13]:


df[varc].value_counts()


# ### Cambio de espacios

# In[14]:


X = df[varc].sample(1_000,random_state=0).copy()


# In[15]:


df_sample = df.sample(1_000,random_state=0).copy()


# ### PCA $\mathcal{X}\to\mathcal{X}_p$

# In[16]:


sc = StandardScaler()
pca = PCA(n_components=3)
Xp = pd.DataFrame(pca.fit_transform(sc.fit_transform(X)))
print(pca.explained_variance_ratio_.cumsum())
Xp


# ### MDS $\mathcal{X}\to\mathcal{X}_m$

# In[17]:


sc = MinMaxScaler()
mds = MDS(n_components=3,n_jobs=-1)
Xm = pd.DataFrame(mds.fit_transform(sc.fit_transform(X)))
Xm


# ### t-SNE $\mathcal{X}\to\mathcal{X}_t$

# In[18]:


sc = MinMaxScaler()
tsne = TSNE(n_components=3,n_jobs=-1,perplexity=15)
Xt = pd.DataFrame(tsne.fit_transform(sc.fit_transform(X)))
Xt


# ### Visualización preeliminar

# ### Vectores

# In[19]:


Xp.iplot(kind='scatter3d',x=0,y=1,z=2,mode='markers',color='purple')


# In[20]:


Xm.iplot(kind='scatter3d',x=0,y=1,z=2,mode='markers',color='purple')


# In[21]:


Xt.iplot(kind='scatter3d',x=0,y=1,z=2,mode='markers',color='purple')


# ### Densidad

# In[22]:


sns.kdeplot(data=Xp,x=0,y=1,fill=True)


# In[23]:


sns.kdeplot(data=Xm,x=0,y=1,fill=True)


# In[24]:


sns.kdeplot(data=Xt,x=0,y=1,fill=True)


# ## Clustering

# In[25]:


sc = MinMaxScaler()
Xs = pd.DataFrame(sc.fit_transform(X),columns=varc)


# In[26]:


Xs


# ### Aglomerativo 

# In[27]:


sil = pd.DataFrame(map(lambda k:(k,silhouette_score(Xs,
                                              AgglomerativeClustering(n_clusters=k).fit_predict(Xs))),
                 range(2,10)),columns=['k','sil'])
plt.plot(sil['k'],sil['sil'],marker='o')


# In[28]:


k = 6
tipo = 'agg'
agg = AgglomerativeClustering(n_clusters=k)
df_sample[f'cl_{tipo}']=Xp[f'cl_{tipo}']=Xm[f'cl_{tipo}']=Xt[f'cl_{tipo}'] =agg.fit_predict(Xs[varc])


# ### K-medias

# In[29]:


sil = pd.DataFrame(map(lambda k:(k,silhouette_score(Xs,
                                              KMeans(n_clusters=k,max_iter=1000).fit_predict(Xs))),
                 range(2,10)),columns=['k','sil'])
plt.plot(sil['k'],sil['sil'],marker='o')


# In[30]:


k = 6
tipo = 'kme'
kme = KMeans(n_clusters=k,max_iter=1000)
df_sample[f'cl_{tipo}']=Xp[f'cl_{tipo}']=Xm[f'cl_{tipo}']=Xt[f'cl_{tipo}'] =kme.fit_predict(Xs[varc])


# ### Modelos Gaussianos Mixtos

# In[31]:


sil = pd.DataFrame(map(lambda k:(k,silhouette_score(Xs,
                                              GaussianMixture(n_components=k,max_iter=1000).fit_predict(Xs))),
                 range(2,10)),columns=['k','sil'])
plt.plot(sil['k'],sil['sil'],marker='o')


# In[32]:


k = 6
tipo = 'gmm'
gmm = GaussianMixture(n_components=k,max_iter=1000)
df_sample[f'cl_{tipo}']=Xp[f'cl_{tipo}']=Xm[f'cl_{tipo}']=Xt[f'cl_{tipo}'] =gmm.fit_predict(Xs[varc])


# ### Selección final

# In[33]:


varcl = sorted(df_sample.filter(like='cl_'))
for v in varcl:
    Xp[v] = Xp[v].astype(str)
    Xm[v] = Xm[v].astype(str)
    Xt[v] = Xt[v].astype(str)
    df_sample[v] = df_sample[v].astype(str)
    
pd.DataFrame(map(lambda cl:(cl,silhouette_score(Xs,df_sample[cl])),varcl),columns=['cluster','sil']).iplot(kind='bar',categories='cluster')


# Se puede elegir cualquiera, nos quedamos con K-means.

# ### Visualización con cluster 

# In[34]:


Xp.iplot(kind='scatter3d',x=0,y=1,z=2,mode='markers',categories='cl_kme')


# ### Guardando el archivo final

# In[35]:


X_total = df[varc]


# In[44]:


df['cluster']=kme.fit_predict(X_total[varc])


# In[45]:


df['date'] = df['ts_transaction_date'].dt.strftime("%Y-%m")
df_final = df[['date','id_atm','cluster']]
df_final


# In[47]:


df_final.to_csv('HERNANDEZ_GONZALEZ_RICARDO_PARAMONT.csv', index=False, header=False)


# In[ ]:




