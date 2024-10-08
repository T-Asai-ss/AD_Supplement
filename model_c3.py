# Development of in silico models that define the applicability domains of binary classifiers: An ITSv2-defined approach for identifying skin sensitization hazards
# Python code of Model C3

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
from catboost import CatBoost, CatBoostRegressor, CatBoostClassifier
from catboost import Pool
from catboost import cv

# ## input data
# In[2]:
tx = pd.DataFrame(pd.read_csv(filepath_or_buffer="./tx2.csv", encoding="ms932", sep=","))
ty = pd.DataFrame(pd.read_csv(filepath_or_buffer="./ty0123.csv", encoding="ms932", sep=","))
ivx = pd.DataFrame(pd.read_csv(filepath_or_buffer="./ix2.csv", encoding="ms932", sep=","))
ivy = pd.DataFrame(pd.read_csv(filepath_or_buffer="./iy0123.csv", encoding="ms932", sep=","))
evx = pd.DataFrame(pd.read_csv(filepath_or_buffer="./ex2.csv", encoding="ms932", sep=","))
evy = pd.DataFrame(pd.read_csv(filepath_or_buffer="./ey0123.csv", encoding="ms932", sep=","))
train_pool = Pool(tx, ty, cat_features = ['h-CLAT P/N', 'CD86 P/N', 'CD54 P/N', 'DPRA P/N', 'KeratinoSens P/N', 'QSARTB P/N', 'Nearest T_0/F_1', 'Nearest TP_0/TN_1/FP_2/FN_3'])
validate_pool = Pool(ivx, ivy, cat_features = ['h-CLAT P/N', 'CD86 P/N', 'CD54 P/N', 'DPRA P/N', 'KeratinoSens P/N', 'QSARTB P/N', 'Nearest T_0/F_1', 'Nearest TP_0/TN_1/FP_2/FN_3'])

# ## optuna
# In[3]:
pip install optuna
import optuna
from sklearn.metrics import balanced_accuracy_score
def objective(trial):
    params = {                    
        'depth' : trial.suggest_int('depth', 3, 9),                                       
        'learning_rate' : trial.suggest_loguniform('learning_rate', 0.01, 0.3),                                 
        'bagging_temperature' :trial.suggest_loguniform('bagging_temperature', 0.01, 10), 
        'od_type': trial.suggest_categorical('od_type', ['IncToDec', 'Iter']),
        }
    model = CatBoostClassifier(**params, num_boost_round = 1000, early_stopping_rounds= 300)
    model.fit(train_pool,eval_set=validate_pool)
    preds = model.predict(ivx)
    score = balanced_accuracy_score(ivy, preds)
    return score
study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=100)
print(study.best_trial)

# ## catboost
# In[4]:
params={
        'depth': 4, 
        'learning_rate': 0.1311354350924807,
        'bagging_temperature': 9.993419675898252,
        'od_type': 'Iter'}
model = CatBoostClassifier(**params, num_boost_round = 1000, early_stopping_rounds= 300)
model.fit(train_pool,eval_set=validate_pool, plot=True)
y_train_p = model.predict(tx)
y_iv_p = model.predict(ivx)
y_ev_p = model.predict(evx)
from sklearn.metrics import confusion_matrix
Acc_t = confusion_matrix(ty, y_train_p)
Acc_i = confusion_matrix(ivy, y_iv_p)
Acc_e = confusion_matrix(evy, y_ev_p)

# ## output data
# In[5]:
imp = model.feature_importances_
imps = pd.DataFrame(imp)
imps.to_csv("C3imp.csv", encoding="shift_jis")
ytp = pd.DataFrame(y_train_p)
ytp.to_csv("C3tp.csv", encoding="shift_jis")
yivp = pd.DataFrame(y_iv_p)
yivp.to_csv("C3ip.csv", encoding="shift_jis")
yevp = pd.DataFrame(y_ev_p)
yevp.to_csv("C3ep.csv", encoding="shift_jis")
