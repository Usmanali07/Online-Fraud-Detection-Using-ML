import streamlit as st
import pandas as pd
import numpy as np
from pycaret.regression import load_model , predict_model
saved_lr = load_model('rfmodel') 

def predict(model , input_df):
    prediction_df= predict_model(estimator=saved_lr , data = input_df)
    prediction = prediction_df['Label'][0]
    return prediction


def run():
    st.sidebar.image("Untitled.png",caption='Virtual University Of Pakistan')
    st.title("Online Payments Fraud Detection system")
    st.sidebar.subheader('Under the Supervision of Dr.  Mushtaq Hussain ')
    get= st.selectbox("Type",['CASH_OUT','CASH_IN','PAYMENT', 'TRANSFER','DEBIT'])
    if get == 'CASH_OUT':
        Type = 1
    if get == 'CASH_IN':
        Type = 2
    if get == 'PAYMENT':
        Type = 3
    if get == 'TRANSFER':
        Type = 4
    if get == 'DEBIT':
        Type = 5
    amount = st.number_input("Amount" , value = 0)
    oldbalanceOrg= st.number_input("Initial balance before the transaction" , value= 0)
    newbalanceOrig= st.number_input("Customer's balance after the transaction" , value= 0)
    output = ""
    input_dict = {'type': Type , 'amount': amount , 'oldbalanceOrg':oldbalanceOrg,'newbalanceOrig':newbalanceOrig}
    input_df=pd.DataFrame([input_dict])
    st.sidebar.bar_chart(input_df)
    
    if st.button('Predict'):
        output=predict(model=saved_lr , input_df= input_df)
        if output == 1:
            output='You do a Fraud Transaction'
        if output == 0:
            output = 'Your Transaction is Not Fraud'
            
        
        
    st.success(output)

run()
