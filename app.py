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
    st.image("Untitled.png",caption='Virtual University Of Pakistan')

    st.title("Online Payments Fraud Detection system")
    st.subheader('Under the Supervision of Dr.  Mushtaq Hussain ')
    
    Type = st.number_input("Type" , min_value=0 , max_value=4 , value= 1)
    
    amount = st.number_input("Amount" , value= 100)
    oldbalanceOrg= st.number_input("Initial balance before the transaction" , value= 1)
    newbalanceOrig= st.number_input("Customer's balance after the transaction" , value= 1)
    
    output = ""
    input_dict = {'type': Type , 'amount': amount , 'oldbalanceOrg':oldbalanceOrg,'newbalanceOrig':newbalanceOrig}
    input_df=pd.DataFrame([input_dict])
    if st.button('Predict'):
        output=predict(model=saved_lr , input_df= input_df)
        output='Prediction is   '+str(output)
        
    st.success('The output {}'.format(output))

run()
