import streamlit as st
import pandas as pd
import pickle
import numpy as np
from xgboost import XGBRegressor

pipe = pickle.load(open('pipe.pkl','rb'))
train = pickle.load(open('train.pkl','rb'))


st.title('LaptopPricePredictor')

TypeName = st.selectbox(
    'TypeName',
    train['TypeName'].unique())
Processor_Brand = st.selectbox(
    'Processor Brand',
    train['cpu brand'].unique())
Ram = st.selectbox(
    'RAM',
    train['Ram'].value_counts().sort_index().index)
Display_Resolution = st.selectbox(
    'Display Resolution',
    ('2560x1600', '1440x900', '1920x1080', '2880x1800', '1366x768',
       '2304x1440', '3200x1800', '1920x1200', '2256x1504', '3840x2160',
       '2160x1440', '2560x1440', '1600x900', '2736x1824', '2400x1600'))
Gpu_Brand = st.selectbox(
    'Gpu Brand',
    train['GpuBrand'].unique())
Laptop_Brand = st.selectbox(
    'Laptop Brand',
    train['Company'].unique())
Ssd = st.selectbox(
    'SSD',
    train['SSD'].value_counts().sort_index().index)
Hdd = st.selectbox(
    'HDD',
    train['HDD'].value_counts().sort_index().index)
Display_size = st.number_input('Display size')
Operating_System = st.selectbox(
    'Operating System',
    train['OpSys'].unique())
Touchscreen = st.selectbox(
    'Touchscreen',
    ['Yes','No'])
QuadHD = st.selectbox(
    'Quad HD+',
    ['Yes','No'])
Ips = st.selectbox(
    'IPS',
    ['Yes','No'])
Weight = st.number_input('Weight')

if st.button('Predict Price'):
    X_res = int(Display_Resolution.split('x')[0])
    y_res = int(Display_Resolution.split('x')[1])
    ppi = (((X_res**2) + (y_res**2))*0.5)/Display_size
    
    if Touchscreen == 'Yes':
        Touchscreen = 1
    else:
        Touchscreen = 0
    
    if QuadHD == 'Yes':
        QuadHD = 1
    else:
        QuadHD = 0

    if Ips == 'Yes':
        Ips = 1
    else:
        Ips = 0
    
    Features = [Laptop_Brand, TypeName, Ram, Operating_System, Weight, Touchscreen, Ips, QuadHD, ppi, Processor_Brand, Ssd, Hdd, Gpu_Brand]
    Query = np.array(Features, dtype=object)
    Query = Query.reshape(1,13)

    st.title('Predicted Price')
    st.title(int(np.exp(pipe.predict(Query))))


