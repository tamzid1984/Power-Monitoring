import numpy as np
import pandas as pd

import pickle
import streamlit as st
from PIL import Image

import warnings
warnings.filterwarnings('ignore')
pkl_filename = "model.pkl"
model = pickle.load(open(pkl_filename, 'rb'))

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


