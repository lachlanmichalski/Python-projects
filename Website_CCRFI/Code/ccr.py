#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 21 20:28:44 2021

@author: lockiemichalski
"""

import numpy as np
import pandas as pd
from flask import Flask, request, jsonify, render_template
import pickle

app = Flask(__name__)
model_glb = pickle.load(open('GLB_RF10feat_model.pkl', 'rb'))
model_us = pickle.load(open('US_RF10feat_model.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('startpage.html')

@app.route('/glb_home')
def glb_home():
    return render_template('index_glb.html')

@app.route('/us_home')
def us_home():
    return render_template('index_us.html')

@app.route('/glb_predict',methods=['POST'])
def predict_glb():

    float_features = [x for x in request.form.values()]
    final_features = [np.array(float_features)]
    prediction = model_glb.predict(final_features)
    prediction = round(prediction[0], 2)

    #prediction probabilities of class
    class_probs = model_glb.predict_proba(final_features)
    #idx = (-class_probs).argsort()[:3] # sorting np array
    
    #values to S&P ratings dict (same as US)
    dict_rating_probs = {0:'AAA', 1:'AA+', 2:'AA', 3:'AA-', 4:'A+', 5:'A', 
                           6:'A-', 7:'BBB+', 8:'BBB', 9:'BBB-', 10:'BB+',
                           11:'BB',12:'BB-', 13:'B+', 14:'B', 15:'B-', 
                           16:'CCC+', 17:'CCC', 18:'CCC-', 19:'CC', 20:'SD', 21:'D'}
    
    #change col names S&P rating value
    class_p_df = pd.DataFrame(class_probs[0]).T
    class_p_df.columns=list(dict_rating_probs.values())
    class_p_df = class_p_df*100 #prob to %
    class_p_df = class_p_df.astype(int)
    
    #same as US        
    dict_rating = {1:'AAA',2:'AA+',3:'AA',4:'AA-',5:'A+',6:'A',7:'A-',
                   8:'BBB+',9:'BBB', 10:'BBB-',11:'BB+',12:'BB',13:'BB-',
                   14:'B+',15:'B',16:'B-',17:'CCC+',18:'CCC',
                   19:'CCC-',20:'CC',23:'SD',24:'D'}
    
    dict_grade = {1:'Investment-grade',2:'Investment-grade',3:'Investment-grade',
                  4:'Investment-grade',5:'Investment-grade',6:'Investment-grade',
                  7:'Investment-grade',8:'Investment-grade',9:'Investment-grade',
                  10:'Investment-grade',11:'High-yield',12:'High-yield',
                  13:'High-yield', 14:'High-yield',15:'High-yield',
                  16:'High-yield',17:'High-yield',18:'High-yield',
                  19:'High-yield',20:'High-yield',23:'High-yield',24:'High-yield'}
        
    output = dict_rating[prediction]
    grade = dict_grade[prediction]

    return render_template("glb_prediction.html", 
                           CTRY = float_features[0],
                           GICS = float_features[1],
                           ICR = float_features[2],
                           IATD = float_features[3],
                           GPM = float_features[4],
                           TLTTA = float_features[5],
                           TLTA = float_features[6],
                           IT = float_features[7],
                           ICA = float_features[8],
                           RT = float_features[9],           
                           prediction_text='Predicted S&P corporate credit rating is {} ({})'.format(output, grade),
                           AAA_prob = str(class_p_df['AAA'][0])+'%',
                           AA_plus_prob = str(class_p_df['AA+'][0])+'%',
                           AA_prob = str(class_p_df['AA'][0])+'%',
                           AA_minus_prob = str(class_p_df['AA-'][0])+'%',
                           A_plus_prob = str(class_p_df['A+'][0])+'%',
                           A_prob = str(class_p_df['A'][0])+'%',
                           A_minus_prob = str(class_p_df['A-'][0])+'%',
                           BBB_plus_prob = str(class_p_df['BBB+'][0])+'%',
                           BBB_prob = str(class_p_df['BBB'][0])+'%',
                           BBB_minus_prob = str(class_p_df['BBB-'][0])+'%',
                           BB_plus_prob = str(class_p_df['BB+'][0])+'%',
                           BB_prob = str(class_p_df['BB'][0])+'%',
                           BB_minus_prob = str(class_p_df['BB-'][0])+'%',
                           B_plus_prob = str(class_p_df['B+'][0])+'%',
                           B_prob = str(class_p_df['B'][0])+'%',
                           B_minus_prob = str(class_p_df['B-'][0])+'%',
                           CCC_plus_prob = str(class_p_df['CCC+'][0])+'%',
                           CCC_prob = str(class_p_df['CCC'][0])+'%',
                           CCC_minus_prob = str(class_p_df['CCC-'][0])+'%',
                           CC_prob = str(class_p_df['CC'][0])+'%',
                           SD_prob = str(class_p_df['SD'][0])+'%',
                           D_prob = str(class_p_df['D'][0])+'%'
                           )

@app.route('/us_predict',methods=['POST'])
def predict_us():

    float_features = [x for x in request.form.values()]
    final_features = [np.array(float_features)]
    prediction = model_us.predict(final_features)
    prediction = round(prediction[0], 2)

    #prediction probabilities of class
    class_probs = model_us.predict_proba(final_features)
    #idx = (-class_probs).argsort()[:3] # sorting np array
    
    #values to S&P ratings dict (same as US)
    dict_rating_probs = {0:'AAA', 1:'AA+', 2:'AA', 3:'AA-', 4:'A+', 5:'A', 
                           6:'A-', 7:'BBB+', 8:'BBB', 9:'BBB-', 10:'BB+',
                           11:'BB',12:'BB-', 13:'B+', 14:'B', 15:'B-', 
                           16:'CCC+', 17:'CCC', 18:'CCC-', 19:'CC', 20:'SD', 21:'D'}
    
    #change col names S&P rating value
    class_p_df = pd.DataFrame(class_probs[0]).T
    class_p_df.columns=list(dict_rating_probs.values())
    class_p_df = class_p_df*100 #prob to %
    class_p_df = class_p_df.astype(int)
    
    #same as US        
    dict_rating = {1:'AAA',2:'AA+',3:'AA',4:'AA-',5:'A+',6:'A',7:'A-',
                   8:'BBB+',9:'BBB', 10:'BBB-',11:'BB+',12:'BB',13:'BB-',
                   14:'B+',15:'B',16:'B-',17:'CCC+',18:'CCC',
                   19:'CCC-',20:'CC',23:'SD',24:'D'}
    
    dict_grade = {1:'Investment-grade',2:'Investment-grade',3:'Investment-grade',
                  4:'Investment-grade',5:'Investment-grade',6:'Investment-grade',
                  7:'Investment-grade',8:'Investment-grade',9:'Investment-grade',
                  10:'Investment-grade',11:'High-yield',12:'High-yield',
                  13:'High-yield', 14:'High-yield',15:'High-yield',
                  16:'High-yield',17:'High-yield',18:'High-yield',
                  19:'High-yield',20:'High-yield',23:'High-yield',24:'High-yield'}
        
    output = dict_rating[prediction]
    grade = dict_grade[prediction]

    return render_template("us_prediction.html", 
                           ICR = float_features[0],
                           LTDTL = float_features[1],
                           IT = float_features[2],
                           GICS = float_features[3],
                           IATD = float_features[4],
                           TLTTA = float_features[5],
                           OPMBD = float_features[6],
                           STDTD = float_features[7],
                           TDTA = float_features[8],
                           ICA = float_features[9],           
                           prediction_text='Predicted S&P corporate credit rating is {} ({})'.format(output, grade),
                           AAA_prob = str(class_p_df['AAA'][0])+'%',
                           AA_plus_prob = str(class_p_df['AA+'][0])+'%',
                           AA_prob = str(class_p_df['AA'][0])+'%',
                           AA_minus_prob = str(class_p_df['AA-'][0])+'%',
                           A_plus_prob = str(class_p_df['A+'][0])+'%',
                           A_prob = str(class_p_df['A'][0])+'%',
                           A_minus_prob = str(class_p_df['A-'][0])+'%',
                           BBB_plus_prob = str(class_p_df['BBB+'][0])+'%',
                           BBB_prob = str(class_p_df['BBB'][0])+'%',
                           BBB_minus_prob = str(class_p_df['BBB-'][0])+'%',
                           BB_plus_prob = str(class_p_df['BB+'][0])+'%',
                           BB_prob = str(class_p_df['BB'][0])+'%',
                           BB_minus_prob = str(class_p_df['BB-'][0])+'%',
                           B_plus_prob = str(class_p_df['B+'][0])+'%',
                           B_prob = str(class_p_df['B'][0])+'%',
                           B_minus_prob = str(class_p_df['B-'][0])+'%',
                           CCC_plus_prob = str(class_p_df['CCC+'][0])+'%',
                           CCC_prob = str(class_p_df['CCC'][0])+'%',
                           CCC_minus_prob = str(class_p_df['CCC-'][0])+'%',
                           CC_prob = str(class_p_df['CC'][0])+'%',
                           SD_prob = str(class_p_df['SD'][0])+'%',
                           D_prob = str(class_p_df['D'][0])+'%'
                           )

if __name__ == "__main__":
    app.run(debug=True)
    