from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import joblib
import os

app = FastAPI(title="House Price Prediction API")

# Data model for house features
class HouseFeatures(BaseModel):
    area: float
    bedrooms: int
    age: int

# Load and train model
def train_model():
    # Read the dataset
    df = pd.read_csv('house_data.csv')
    
    # Prepare features and target
    X = df[['Area', 'Bedrooms', 'Age']]
    y = df['Price']
    
    # Train the model
    model = LinearRegression()
    model.fit(X, y)
    
    return model

# Initialize model
model = train_model()

@app.get("/")
def read_root():
    return {"message": "House Price Prediction API", 
            "usage": "Send POST request to /predict with area, bedrooms, and age"}

@app.post("/predict")
def predict_price(features: HouseFeatures):
    try:
        # Prepare input features
        input_data = pd.DataFrame({
            'Area': [features.area],
            'Bedrooms': [features.bedrooms],
            'Age': [features.age]
        })
        
        # Make prediction
        prediction = model.predict(input_data)[0]
        
        return {
            "predicted_price": round(prediction, 2),
            "features": features.dict()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/model-info")
def model_info():
    return {
        "model_type": "Linear Regression",
        "features": ["Area", "Bedrooms", "Age"],
        "target": "Price"
    } 