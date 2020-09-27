#!/usr/bin/env python
# coding: utf-8

# In[1]:


import warnings
import pandas as pd
import numpy as np 
from sklearn.impute import SimpleImputer
from pandas.plotting import scatter_matrix
from scipy import stats
import matplotlib.pyplot as plt
from sklearn import model_selection
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
import seaborn as sns
from scipy.stats.mstats import winsorize
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
import statsmodels.api as sm
from sklearn.model_selection import train_test_split as split
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score


# In[2]:


warnings.simplefilter(action='ignore', category=FutureWarning)
get_ipython().run_line_magic('matplotlib', 'inline')
sns.set(style="whitegrid", color_codes=True)


# In[3]:


#Read CSV Data
dataframe=pd.read_csv("http://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data")
subset=dataframe[:100000]
dataframe.head(1)


# In[4]:


#Change Coloumn name
dataframe.columns = ('age','sex','cp','trestbps','chol','fbs','restecg','thalach','exang','oldpeak','slope','ca','thal','num')
#Replace '?' with Not a Number "NaN"
dataframe = dataframe.replace('?','NaN')
print(dataframe.tail())
print(dataframe.shape)


# In[5]:


dataframe['ca']=dataframe['ca'].astype(float)
dataframe['thal']=dataframe['thal'].astype(float)

#change the NaN value to its respective Column Mean
imp = SimpleImputer(missing_values = 'NaN', strategy = 'mean')
df_m=dataframe.fillna(dataframe.mean())
df_m.tail(3)


# In[6]:


df_m['ca'] = pd.to_numeric(df_m['ca'], errors='coerce')
df_m[['age', 'sex', 'fbs', 'exang', 'ca']] = df_m[['age', 'sex', 'fbs', 'exang', 'ca']].astype(int)
df_m[['trestbps', 'chol', 'thalach', 'oldpeak']] = df_m[['trestbps', 'chol', 'thalach', 'oldpeak']].astype(float)
df_m['num'].replace(to_replace=[1, 2, 3, 4], value=1, inplace=True)


# In[7]:


sns.boxplot(x=df_m['ca'])


# In[8]:


df_m['Ca']=winsorize(df_m['ca'],limits=[0.0,0.25])
df_m.drop("ca", axis=1, inplace=True) 
sns.boxplot(x=df_m['Ca'])


# In[9]:


sns.boxplot(x=df_m['chol'])


# In[10]:


df_m['Chol']=winsorize(df_m['chol'],limits=[0.0,0.25])
sns.boxplot(x=df_m['Chol'])
df_m.drop("chol", axis=1, inplace=True) 


# In[11]:


sns.boxplot(x=df_m['oldpeak'])


# In[12]:


df_m['Oldpeak']=winsorize(df_m['oldpeak'],limits=[0.03,0.05])
sns.boxplot(x=df_m['Oldpeak'])
df_m.drop("oldpeak", axis=1, inplace=True) 


# In[13]:


#Box Plot
sns.boxplot(x=df_m['trestbps'])


# In[14]:


# Winsorization
df_m['Trestbps']=winsorize(df_m['trestbps'],limits=[0.0,0.25])
sns.boxplot(x=df_m['Trestbps'])
df_m.drop("trestbps", axis=1, inplace=True) 


# In[15]:


sns.boxplot(x=df_m['thal'])


# In[16]:


df_m['Thal']=winsorize(df_m['thal'],limits=[0.03,0.05])
sns.boxplot(x=df_m['Thal'])
df_m.drop("thal", axis=1, inplace=True) 


# In[17]:


sns.boxplot(x=df_m['thalach'])


# In[18]:


df_m['Thalach']=winsorize(df_m['thalach'],limits=[0.03,0.05])
sns.boxplot(x=df_m['Thalach'])
df_m.drop("thalach", axis=1, inplace=True) 


# In[19]:


df_m.head(1)


# In[20]:


#Zscore
z = np.abs(stats.zscore(df_m))
print(z)


# In[21]:


threshold = 3
print(np.where(z > 3))


# In[22]:


#Remove Z Score Outliers
df_m =df_m[(z < 3).all(axis=1)]
df_m.shape


# In[23]:


corr=df_m.corr()
sns.heatmap(corr,
           xticklabels=corr.columns.values,
           yticklabels=corr.columns.values)


# In[24]:


cp_dummy = pd.get_dummies(df_m['cp'])
cp_dummy.rename(columns={1: 'cp_typical_angina', 2: 'cp_atypical_angina',
                         3: 'cp_non_angina', 4: 'cp_asymptomatic_angina'}, inplace=True)
restecg_dummy = pd.get_dummies(df_m['restecg'])
restecg_dummy.rename(columns={0: 'restecg_normal', 1: 'restecg_wave_abnorm',
                              2: 'restecg_ventricular_ht'}, inplace=True)
slope_dummy = pd.get_dummies(df_m['slope'])
slope_dummy.rename(columns={1: 'slope_upsloping', 2: 'slope_flat',
                            3: 'slope_downsloping'}, inplace=True)
thal_dummy = pd.get_dummies(df_m['Thal'])
thal_dummy.rename(columns={3: 'thal_normal', 6: 'thal_fixed_defect',
                           7: 'thal_reversible_defect'}, inplace=True)
df_m = pd.concat([df_m, cp_dummy, restecg_dummy, slope_dummy, thal_dummy], axis=1)

df_m.drop(['cp', 'restecg', 'slope', 'Thal'], axis=1, inplace=True)
df_m.head(2)


# In[62]:


df_X = df_m.drop('num', axis=1)
df_y = df_m['num']

selected_features = []
rfe = RFE(LogisticRegression())

rfe.fit(df_X.values, df_y.values)

for i, feature in enumerate(df_X.columns.values):
    if rfe.support_[i]:
        selected_features.append(feature)

selected_X = df_X[selected_features]
selected_y = df_y

lm = sm.Logit(selected_y, selected_X)
result = lm.fit()

selected_X_train, selected_X_test, selected_y_train, selected_y_test = split(selected_X, selected_y, test_size=0.3, random_state=0)
lr = LogisticRegression()
lr.fit(selected_X_train, selected_y_train)
auc_unprocessed=(selected_X_train, selected_y_train,selected_X_test,selected_y_test )


print(f"Accuracy: {lr.score(selected_X_test, selected_y_test):0.3f}")


# In[63]:


# predict probabilities
probs = lr.predict_proba(selected_X_test)
# keep probabilities for the positive outcome only
probs = probs[:, 1]
# calculate AUC
auc = roc_auc_score(selected_y_test, probs)

# calculate roc curve
fpr, tpr, thresholds = roc_curve(selected_y_test, probs)
# plot no skill
plt.plot([0, 1], [0, 1], linestyle='--')
# plot the roc curve for the model
plt.plot(fpr, tpr, marker='*')
# show the plot
plt.show()


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




