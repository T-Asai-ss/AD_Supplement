# Development of in silico models that define the applicability domains of binary classifiers: An ITSv2-defined approach for identifying skin sensitization hazards
# Python code of ROC

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
from sklearn import metrics

# ## input data
# In[2]:
rmiv = pd.DataFrame(pd.read_csv(filepath_or_buffer="./rmti2.csv", encoding="ms932", sep=","))
ivy = pd.DataFrame(pd.read_csv(filepath_or_buffer="./yti.csv", encoding="ms932", sep=","))

# ## roc
# In[3]:
fpr, tpr, thres = metrics.roc_curve(ivy, rmiv)
auc = metrics.auc(fpr, tpr)
print('auc:', auc)

Youden_index_candidates = tpr-fpr    
index = np.where(Youden_index_candidates==max(Youden_index_candidates))[0][0]

# ## plot
# In[4]:
plt.plot(fpr, tpr, label='ROC curve (area = %.2f)'%auc)
plt.legend()
plt.title('ROC curve')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.grid(True)

# ## output data
# In[5]:
ROC_df = pd.DataFrame({'fpr':fpr, 'tpr':tpr, 'thresholds':thres})
