#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 20 13:57:06 2021

@author: lockiemichalski
"""

import pandas as pd
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pickle


################################################################################
# BUILD 10 FEATURE RF FOR US SAMPLE
################################################################################

#uncomment the """ .... """ if wish to run the results. The output is in xlsx file.
   
'''set directory for where functions for are data and import in mda_recursive.py'''

os.chdir('')

us_data = pd.read_csv('US_NONESG_SAMPLE_CLEAN_USE_SOV.csv').iloc[:,1:] # all classes

data = us_data[['rating_rank','intcov_ratioq','lt_debtq','inv_turnq','gind',
                   'int_totdebtq','lt_ppentq','opmbdq','short_debtq','debt_atq',
                   'invt_actq']]

'''Get full data to train and test model for mda values'''
max_y = max(data.iloc[:,0]) # determine if data is binary or mc

# split data into X and y (y has to be first col)
Xdata = data.iloc[:,1:]
ydata = data.iloc[:,0]

# split data into train and test sets
X_train_mda, X_test_mda, y_train_mda, y_test_mda = train_test_split(Xdata, ydata, random_state=42)
    
model_mda = RandomForestClassifier(random_state=0)
model_test = RandomForestClassifier(random_state=0)

'''cv training, validation metrics'''
#k_folds = StratifiedKFold(n_splits=5)
#splits = list(k_folds.split(X_train_mda, y_train_mda))
#cv_acc = cross_val_score(model_mda, X_train_mda, y_train_mda, cv=splits, scoring='accuracy')
#cv_acc_dict[str(num)] = cv_acc.mean().round(4)
#cv_acc_std_dict[str(num)] = cv_acc.std().round(4)

#cv_auc=cross_val_score(model_mda, X_train_mda, y_train_mda, cv=splits, scoring='roc_auc_ovr_weighted')
#cv_roc_dict[str(num)] = cv_auc.mean().round(4)
#cv_roc_std_dict[str(num)] = cv_auc.std().round(4)

'''testing set metrics'''
model_test.fit(X_train_mda, y_train_mda)
y_pred = model_test.predict(X_test_mda)
predictions = [round(value) for value in y_pred]
accuracy = accuracy_score(y_test_mda, predictions)
#acc_test_dict[str(num)]=accuracy.round(4)
#ynew = model_test.predict_proba(X_test_mda)

################################################################################
# PICKLE MODEL
################################################################################

os.chdir('/Users/lockiemichalski/Documents/UQ/Credit Research/US')

#Serializing the model
with open('US_RF10feat_model.pkl', 'wb') as f:
    pickle.dump(model_test, f)

#De-Serializing the model
with open('US_RF10feat_model.pkl', 'rb') as f:
    clf_loaded = pickle.load(f)
    
#Check the pickle file by inputing the variables
model = pickle.load(open('US_RF10feat_model.pkl','rb'))

#final_features = [np.array([2, 0.4, 7,252010, 0.083, 4.2, 0.991, 0.040, 0.33, 0.001])]
#prediction = model.predict(final_features)
#output = round(prediction[0], 2)

################################################################################
# UPLOAD files to digitalocean
################################################################################

