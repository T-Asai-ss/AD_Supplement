# Development of in silico models that define the applicability domains of binary classifiers: An ITSv2-defined approach for identifying skin sensitization hazards
# Python code of Molecular Descriptor selection using Boruta

# ## import library
# In[1]:
import numpy as np
import pandas as pd
import csv
import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")
from sklearn.ensemble import RandomForestRegressor
from boruta import BorutaPy

# ## input data
# In[2]:
tx = pd.DataFrame(pd.read_csv(filepath_or_buffer="./txdes.csv", encoding="ms932", sep=","))
ty = pd.DataFrame(pd.read_csv(filepath_or_buffer="./ty01.csv", encoding="ms932", sep=","))
ivx = pd.DataFrame(pd.read_csv(filepath_or_buffer="./ixdes.csv", encoding="ms932", sep=","))
ivy = pd.DataFrame(pd.read_csv(filepath_or_buffer="./iy01.csv", encoding="ms932", sep=","))
evx = pd.DataFrame(pd.read_csv(filepath_or_buffer="./exdes.csv", encoding="ms932", sep=","))
evy = pd.DataFrame(pd.read_csv(filepath_or_buffer="./ey01.csv", encoding="ms932", sep=","))

# ## boruta
# In[3]:
model = RandomForestRegressor(n_jobs=-1, max_depth=5)
feat_selector = BorutaPy(model, n_estimators='auto', two_step=False, perc=85, verbose=3, random_state=42)
feat_selector.fit(tx.values, ty.values)
selected = feat_selector.support_

# ## output data
# In[4]:
selected = pd.DataFrame(selected)
selected.to_csv("moldesBoruta.csv", encoding="shift_jis")
