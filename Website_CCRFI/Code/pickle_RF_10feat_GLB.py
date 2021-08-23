#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 20 13:57:06 2021

@author: lockiemichalski
"""

import pandas as pd
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pickle

################################################################################
# BUILD 14 FEATURE RF FOR GLB SAMPLE
################################################################################

#uncomment the """ .... """ if wish to run the results. The output is in xlsx file.
   
'''set directory for where functions for are data and import in mda_recursive.py'''

os.chdir('/Users/lockiemichalski/Documents/UQ/Credit Research')

gl_data = pd.read_csv('GLOBAL_NONESG_SAMPLE_CLEAN_FULL_SOV.csv').iloc[:,1:] # all classes
gl_data = gl_data[['Country of Exchange', 'loc_rank']]
gl_data.groupby('loc_rank').first().to_csv('countries.csv')

'''GLB'''
gl_data = pd.read_csv('GLOBAL_NONESG_SAMPLE_CLEAN_USE_SOV.csv').iloc[:,1:] # all classes

data = gl_data[['rating_rank','loc_rank','gind','intcov_ratioq','int_totdebtq',
                'gpmq','lt_ppentq','lt_atq','inv_turnq','invt_actq','rect_turnq']]


'''Country - loc_rank
GICS Industry - gind
Interest Coverage Ratio - intcov_ratioq
Interest/Average Total Debt - int_totdebtq
Gross Profit Margin - gpmq
Total Liabilities/Total Tangible Assets - lt_ppentq
Total Liabilities/Total Assets - lt_atq
Inventory Turnover - inv_turnq
Inventory/Current Assets - invt_actq
Receivables Turnover = rect_turnq'''


'''Get full data to train and test model for mda values'''
max_y = max(data.iloc[:,0]) # determine if data is binary or mc

# split data into X and y (y has to be first col)
Xdata = data.iloc[:,1:]
ydata = data.iloc[:,0]

# split data into train and test sets
X_train_mda, X_test_mda, y_train_mda, y_test_mda = train_test_split(Xdata, ydata, random_state=42)
    
model_test = RandomForestClassifier(random_state=0, n_estimators=100)

'''testing set metrics'''
model_test.fit(X_train_mda, y_train_mda)
y_pred = model_test.predict(X_test_mda)
predictions = [round(value) for value in y_pred]
accuracy = accuracy_score(y_test_mda, predictions)

################################################################################
# PICKLE MODEL
################################################################################

os.chdir('/Users/lockiemichalski/Documents/UQ/Credit Research/Global')

#Serializing the model
with open('GLB_RF10feat_model.pkl', 'wb') as f:
    pickle.dump(model_test, f)

#De-Serializing the model
with open('GLB_RF10feat_model.pkl', 'rb') as f:
    clf_loaded = pickle.load(f)
    
#Check the pickle file by inputing the variables
model = pickle.load(open('GLB_RF10feat_model.pkl','rb'))

#final_features = [np.array([2, 0.4, 7,252010, 0.083, 4.2, 0.991, 0.040, 0.33, 0.001])]
#prediction = model.predict(final_features)
#output = round(prediction[0], 2)

################################################################################
# UPLOAD files to digitalocean
################################################################################
#scp -r /Users/lockiemichalski/Desktop/Webpage/US_MDA  root@138.68.170.171:~


