# Development of in silico models that define the applicability domains of binary classifiers: An ITSv2-defined approach for identifying skin sensitization hazards
# Python code of Model R2

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
txsmgn =  pd.DataFrame(pd.read_csv(filepath_or_buffer="./tbsmgn2.csv", encoding="ms932", sep=","))
tx = pd.DataFrame(pd.read_csv(filepath_or_buffer="./tx2.csv", encoding="ms932", sep=","))
ty = pd.DataFrame(pd.read_csv(filepath_or_buffer="./tyb.csv", encoding="ms932", sep=","))
ivx = pd.DataFrame(pd.read_csv(filepath_or_buffer="./ix2.csv", encoding="ms932", sep=","))
ivy = pd.DataFrame(pd.read_csv(filepath_or_buffer="./iyb.csv", encoding="ms932", sep=","))
evx = pd.DataFrame(pd.read_csv(filepath_or_buffer="./ex2.csv", encoding="ms932", sep=","))
evy = pd.DataFrame(pd.read_csv(filepath_or_buffer="./eyb.csv", encoding="ms932", sep=","))

# ## smote
# In[3]:
pip install smogn
import smogn
rg_mtrx = [
    [0, 0, 0],
    [0.333, 1, 0],
    [0.667, 1, 0],
    [1, 0, 0]
]
smgn = smogn.smoter(
    data = txsmgn,
    y = 'Yb',
    k = 3,
    rel_thres = 0.80,
    rel_method = 'manual',
    rel_ctrl_pts_rg = rg_mtrx)
tysm = smgn['Yb']
txsm = smgn.drop("Yb", axis=1)
tysmdf = pd.DataFrame(tysm.T)
smo = pd.DataFrame(smgn)
smo.to_csv("R2smgn.csv", encoding="shift_jis")
txsm = pd.DataFrame(pd.read_csv(filepath_or_buffer="./txsmR2.csv", encoding="ms932", sep=","))
tysm = pd.DataFrame(pd.read_csv(filepath_or_buffer="./tysmR2.csv", encoding="ms932", sep=","))
train_pool = Pool(txsm, tysm, cat_features = ['h-CLAT P/N', 'CD86 P/N', 'CD54 P/N', 'DPRA P/N', 'KeratinoSens P/N', 'QSARTB P/N', 'Nearest T_0/F_1', 'Nearest TP_0/TN_1/FP_2/FN_3'])
validate_pool = Pool(ivx, ivy, cat_features = ['h-CLAT P/N', 'CD86 P/N', 'CD54 P/N', 'DPRA P/N', 'KeratinoSens P/N', 'QSARTB P/N', 'Nearest T_0/F_1', 'Nearest TP_0/TN_1/FP_2/FN_3'])

# ## optuna
# In[4]:
pip install optuna
import optuna
from sklearn.metrics import r2_score
def objective(trial):
    params = {             
        'depth' : trial.suggest_int('depth', 3, 9),
        'learning_rate' : trial.suggest_loguniform('learning_rate', 0.01, 0.3),                                 
        'bagging_temperature' :trial.suggest_loguniform('bagging_temperature', 0.01, 10), 
        'od_type': trial.suggest_categorical('od_type', ['IncToDec', 'Iter']),
        }
    model = CatBoostRegressor(**params, num_boost_round = 1000, early_stopping_rounds= 300)
    model.fit(train_pool,eval_set=validate_pool)
    preds = model.predict(ivx)
    score = r2_score(ivy, preds)
    return score
study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=100)
print(study.best_trial)

# ## catboost
# In[5]:
params={
        'depth': 4,
        'learning_rate': 0.1961625733494209,
        'bagging_temperature': 0.024560839854123974,
        'od_type': 'IncToDec',
        }
model = CatBoostRegressor(**params, num_boost_round = 1000, early_stopping_rounds= 300)
model.fit(train_pool,eval_set=validate_pool, plot=True)
y_train_p = model.predict(tx)
y_iv_p = model.predict(ivx)
y_ev_p = model.predict(evx)
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
print('R^2  train: %.3f,  internal validation: %.3f, external validation: %.3f' % (
        r2_score(ty, y_train_p),
        r2_score(ivy, y_iv_p),
        r2_score(evy, y_ev_p)))
print('RMSE  train: %.3f,  internal validation: %.3f, external validation: %.3f' %(
        np.sqrt(mean_squared_error(ty, y_train_p)),
        np.sqrt(mean_squared_error(ivy, y_iv_p)),
        np.sqrt(mean_squared_error(evy, y_ev_p))))

# ## plot
# In[6]:
fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
ax.scatter(ty, y_train_p, s=30, marker='o', alpha=1, c='w', edgecolors='k', linewidths=1.5)
ax.scatter(ivy, y_iv_p, s=30, marker='x', alpha=1, c='k', edgecolors='k', linewidths=1.5)
ax.scatter(evy, y_ev_p, s=30, marker='^', alpha=1, c='w', edgecolors='k', linewidths=1.5)
ax.set_xlim(left=-0.5, right=1.5) 
ax.set_ylim(bottom=-0.5, top=1.5)
ax.set_xlabel('Published Log(EC3(µmol/cm$^2$))')
ax.set_ylabel('Predicted Log(EC3(µmol/cm$^2$))')
ax.set_title("Model R2")
ax.legend(['Training', 'Internal-valid.','External-valid.'])
plt.show()       

# ## output data
# In[7]:
imp = model.feature_importances_
imps = pd.DataFrame(imp)
imps.to_csv("R2imp.csv", encoding="shift_jis")
ytp = pd.DataFrame(y_train_p)
ytp.to_csv("R2tp.csv", encoding="shift_jis")
yivp = pd.DataFrame(y_iv_p)
yivp.to_csv("R2ip.csv", encoding="shift_jis")
yevp = pd.DataFrame(y_ev_p)
yevp.to_csv("R2ep.csv", encoding="shift_jis")
