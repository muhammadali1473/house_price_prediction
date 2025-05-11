import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import joblib

# Set page configuration
st.set_page_config(
    page_title="House Price Prediction",
    page_icon="🏠",
    layout="centered"
)

# Add custom CSS
st.markdown("""
    <style>
    .main {
        padding: 2rem;
    }
    .stButton>button {
        width: 100%;
        background-color: #4CAF50;
        color: white;
    }
    </style>
    """, unsafe_allow_html=True)

# Title and description
st.title("🏠 House Price Prediction")
st.markdown("### Predict house prices based on area, bedrooms, and age")

# Load and train model
@st.cache_data
def load_and_train_model():
    # Read the dataset
    df = pd.read_csv('house_data.csv')
    
    # Prepare features and target
    X = df[['Area', 'Bedrooms', 'Age']]
    y = df['Price']
    
    # Train the model
    model = LinearRegression()
    model.fit(X, y)
    
    return model, df

# Load model and data
model, df = load_and_train_model()

# Create two columns
col1, col2 = st.columns(2)

with col1:
    # Input features
    st.subheader("House Features")
    area = st.number_input("Area (sq ft)", min_value=100, max_value=10000, value=1500)
    bedrooms = st.number_input("Number of Bedrooms", min_value=1, max_value=10, value=3)
    age = st.number_input("Age of House (years)", min_value=0, max_value=100, value=5)

with col2:
    # Display some statistics
    st.subheader("Dataset Statistics")
    st.write("Average House Price:", f"${df['Price'].mean():,.2f}")
    st.write("Average Area:", f"{df['Area'].mean():,.0f} sq ft")
    st.write("Average Bedrooms:", f"{df['Bedrooms'].mean():.1f}")
    st.write("Average Age:", f"{df['Age'].mean():.1f} years")

# Prediction button
if st.button("Predict Price"):
    # Prepare input data
    input_data = pd.DataFrame({
        'Area': [area],
        'Bedrooms': [bedrooms],
        'Age': [age]
    })
    
    # Make prediction
    prediction = model.predict(input_data)[0]
    
    # Display prediction with nice formatting
    st.success(f"### Predicted House Price: ${prediction:,.2f}")
    
    # Display feature importance
    st.subheader("Feature Importance")
    importance_df = pd.DataFrame({
        'Feature': ['Area', 'Bedrooms', 'Age'],
        'Coefficient': model.coef_
    })
    st.bar_chart(importance_df.set_index('Feature'))

# Add data visualization
st.subheader("Data Visualization")
chart_type = st.selectbox("Select Chart", ["Price vs Area", "Price vs Bedrooms", "Price vs Age"])

if chart_type == "Price vs Area":
    st.scatter_chart(data=df, x='Area', y='Price')
elif chart_type == "Price vs Bedrooms":
    st.scatter_chart(data=df, x='Bedrooms', y='Price')
else:
    st.scatter_chart(data=df, x='Age', y='Price') 