import numpy as np
import pandas as pd
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split  
from sklearn.metrics import accuracy_score 
from sklearn.metrics import confusion_matrix
from sklearn. metrics import classification_report, roc_auc_score, roc_curve
import pickle
import streamlit as st
import matplotlib.pyplot as plt
from sklearn import metrics
import seaborn as sns

import warnings
warnings.filterwarnings('ignore')

model = pickle.load(open(r'model.pkl', 'rb'))

def run():
    # Adding title and Image
    img1 = Image.open('logo.png')
    img1 = img1.resize((156,145))
    st.image(img1,use_column_width=False)
    st.title("Power Monitor system of Energy Meter")

    # Voltage
    Voltage = st.number_input('Voltage')

    # Current
    Current = st.number_input('Current')

    # Power Consumption
    Power = st.number_input('Power')

    

    if st.button('Submit'):
        features = [[Voltage, Current, Power]]
        print(features)
        prediction = model.predict(features)
        st.success(
                f' {(prediction)}'
            )
run()


