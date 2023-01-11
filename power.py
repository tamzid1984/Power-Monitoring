import numpy as np
import pickle
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from pandas.plotting import scatter_matrix
from matplotlib import pyplot
from pandas import read_csv

from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold
import streamlit as st
from PIL import Image
import warnings
warnings.filterwarnings('ignore')
model = pickle.load(open('model.pkl', 'rb'))

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


