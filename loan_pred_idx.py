# import libraries

import numpy as np 
import pandas as pd 
import seaborn as sns
import matplotlib.pyplot as plt

import sklearn.preprocessing as preprocessing
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_curve,auc
from sklearn.metrics import classification_report

from pickle import dump
from pickle import load

# set the max columns to none
pd.set_option('display.max_columns', None)

# read the csv file
link1 = 'path_you_have/loan_data_2007_2014.csv'
df = pd.read_csv(link1)

# drop unuseful columns
no_use_columns = [
    'inq_last_12m', 'total_cu_tl', 'inq_fi', 'open_acc_6m',	
    'open_il_6m',	'open_il_12m',	'open_il_24m',	'mths_since_rcnt_il',	
    'total_bal_il',	'il_util',	'open_rv_12m',	'open_rv_24m',	'max_bal_bc',	
    'all_util', 'annual_inc_joint',	'dti_joint','verification_status_joint',
    'Unnamed: 0', 'zip_code', 'desc', 'url', 'title', 'application_type',
    'issue_d', 'next_pymnt_d',	'last_credit_pull_d', 
    'last_pymnt_d', 'earliest_cr_line'
]

df.drop(no_use_columns, inplace=True, axis=1)

no_use_columns2 = [
    'grade', 'id', 'member_id', 'emp_title'
]

df.drop(no_use_columns2, inplace=True, axis=1)

# Drop column with all null value, and drop some rows with null values
# copy the dataset just in case we need to use the original
df_copy = df.copy()

df_copy.drop('mths_since_last_record',axis=1,inplace=True)
df_copy.drop('mths_since_last_delinq',axis=1,inplace=True)
df_copy.drop('mths_since_last_major_derog',axis=1,inplace=True)

df_clean = df_copy.dropna(axis=0)

# Drop policy_code column, because it only has one value
df_clean.drop('policy_code', inplace=True, axis=1)

# Change label in Status (For classification)
df_clean['loan_status'].replace({'Fully Paid': 0, 'Current' : 0}, inplace=True)
df_clean['loan_status'].replace({'Charged Off': 1, 
                           'Late (31-120 days)' : 1, 
                           'In Grace Period' : 1,
                           'Late (16-30 days)' : 1,
                           'Default' : 1}, inplace=True)
df_clean['loan_status'] = df_clean['loan_status'].astype('int')


# EDA Multivariate
# # Heatmap
# plt.figure(figsize=(22,15))
# corrs = df.corr()
# sns.heatmap(corrs, cmap='RdBu_r', annot=True)
# plt.show()

# Drop columns with highly correlated values, only one is being used
df_use = df_clean.copy()
df_use.drop(columns=['funded_amnt', 'funded_amnt_inv', 
                     'revol_bal', 'out_prncp_inv', 'total_pymnt_inv', 
                     'total_rec_prncp', 'recoveries'
                     ], axis=1, inplace=True)

# Encoding categorical features
# Label encoding using sklearn
labelEnc = preprocessing.LabelEncoder()
for x in df_use:
    if df_use[x].dtypes=='object':
        df_use[x] = labelEnc.fit_transform(df_use[x])


# split to variable feature and target/labels (x and y)
x = df_use.loc[:, df_use.columns != 'loan_status'] # X value contains all the variables except labels
y = df_use['loan_status'] # these are the labels
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3)


# Scaling using Standard Scaller
ss = preprocessing.StandardScaler()
x_train_scaled = pd.DataFrame(ss.fit_transform(x_train), columns=x_train.columns)
x_test_scaled = pd.DataFrame(ss.transform(x_test), columns=x_test.columns)

# Handling Imbalance Data
# Oversampling
oversample = SMOTE()
x_train_balanced, y_train_balanced = oversample.fit_resample(x_train_scaled, y_train)
x_test_balanced, y_test_balanced = oversample.fit_resample(x_test_scaled, y_test)


# Modeling
# Training Phase
rf=RandomForestClassifier()
model_rf = rf.fit(x_train_balanced,y_train_balanced)
train_score = rf.score(x_train_balanced, y_train_balanced)
test_score = rf.score(x_test_balanced, y_test_balanced)

# Predict
pred_rf=rf.predict(x_test_balanced)
probs_rf=rf.predict_proba(x_test_balanced)[:,1]

# classification report
print(classification_report(y_test_balanced, pred_rf))

# Create a function to plot ROC Curves
def plot_roc(y_test,probs):
    fpr,tpr,threshold=roc_curve(y_test_balanced,probs)
    roc_auc=auc(fpr,tpr)
    print('ROC AUC=%0.2f'%roc_auc)
    plt.plot(fpr,tpr,label='AUC=%0.2f'%roc_auc,color='darkorange')
    plt.legend(loc='lower right')
    plt.plot([0,1],[0,1],'b--')
    plt.xlim([0,1])
    plt.ylim([0,1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.show()

#random forest
plot_roc(y_test,probs_rf)

# save the model 
filename = 'path_you_have/finalized_model.sav'
dump(model_rf, open(filename, 'wb'))

