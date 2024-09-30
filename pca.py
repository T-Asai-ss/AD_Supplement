# Development of in silico models that define the applicability domains of binary classifiers: Example of ITSv2 Defined Approach for skin sensitization hazard identification
# Python code of PCA

# ## import library
# In[1]:
import numpy as np
import pandas as pd
import seaborn as sns
import urllib.request 
import matplotlib.pyplot as plt
%matplotlib inline
import sklearn 
from sklearn.decomposition import PCA
import csv

# ## input data
# In[2]:
df = pd.DataFrame(pd.read_csv(filepath_or_buffer="./alldes2.csv", encoding="ms932", sep=","))
dfs = df.iloc[:, 2:].apply(lambda x: (x-x.min())/(x.max()-x.min()), axis=0)

# ## pca
# In[3]:
pca = PCA(n_components=2)
df_pca = pd.DataFrame(pca.fit_transform(dfs))

# ## plot
# In[4]:
label = df['Role']
df_pca['label'] = label
sns.scatterplot(data=df_pca, x=0, y=1, hue='label', style='label')

# ## output data
# In[5]:
labels = pd.DataFrame(label)
labels.to_csv("230808 label.csv", encoding="shift_jis")
pca = pd.DataFrame(df_pca)
pca.to_csv("230808 pca.csv", encoding="shift_jis")
