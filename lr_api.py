
import pandas as pd
from pycaret.classification import load_model, predict_model
from fastapi import FastAPI
import uvicorn

# Create the app
app = FastAPI()

# Load trained Pipeline
model = load_model('lr_api')

# Define predict function
@app.post('/predict')
def predict(type, amount, oldbalanceOrg, newbalanceOrig):
    data = pd.DataFrame([[type, amount, oldbalanceOrg, newbalanceOrig]])
    data.columns = ['type', 'amount', 'oldbalanceOrg', 'newbalanceOrig']
    predictions = predict_model(model, data=data) 
    return {'prediction': list(predictions['Label'])}

if __name__ == '__main__':
    uvicorn.run(app, host='127.0.0.1', port=5000)
