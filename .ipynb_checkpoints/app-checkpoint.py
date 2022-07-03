import streamlit as st
import pandas as pd
import numpy as np
from pycaret.regression import load_model 
saved_lr = load_model('etmodel') 

def predict(model , input_df):
    prediction_df= predict_model(estimator=saved_lr , data = input_df)
    prediction = prediction_df['Label'][0]
    return prediction


def run():
    st.sidebar.info("This app  is created to predict Fraud detection")
    st.sidebar.success("Usman Ali")
    st.title("Fraud Detection")
    
    Type = st.number_input("Type" , min_value=0 , max_value=4 , value= 1)
    amount = st.number_input("amount" , min_value=0 , max_value=100000 , value= 100)
    oldbalanceOrg= st.number_input("oldbalanceOrg" , min_value=0 , max_value=100000, value= 1)
    newbalanceOrg= st.number_input("newbalanceOrig" , min_value=0 , max_value=100000, value= 1)
    
    output = ""
    input_dict = {'type': Type , 'amount': amount , 'oldbalanceOrg':oldbalanceOrg,'newbalanceOrg':newbalanceOrg}
    input_df=pd.DataFrame([input_dict])
    if st.button('Predict'):
        output=predict(model=save_lr , input_df= input_df)
        output='$'+str(output)
        
    st.success('The output is {}'.format(output))