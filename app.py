import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error

# Set page configuration
st.set_page_config(
    page_title="House Price Prediction",
    page_icon="🏠",
    layout="wide"
)

# Add custom CSS
st.markdown("""
    <style>
    .main {
        padding: 20px;
    }
    .stButton>button {
        width: 100%;
        margin-top: 20px;
    }
    </style>
    """, unsafe_allow_html=True)

# Title and description
st.title("🏠 House Price Prediction")
st.markdown("""
This app predicts house prices based on features like area, number of bedrooms, and age of the house.
Enter your house details below to get a prediction!
""")

def load_data():
    return pd.read_csv('house_data.csv')

def train_model(df):
    X = df[['Area', 'Bedrooms', 'Age']]
    y = df['Price']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    return model, X_test, y_test

# Load data and train model
df = load_data()
model, X_test, y_test = train_model(df)

# Sidebar for user input
st.sidebar.header("Enter House Details")

area = st.sidebar.slider("Area (sq ft)", 
                        min_value=1000,
                        max_value=5000,
                        value=2500,
                        step=100)

bedrooms = st.sidebar.slider("Number of Bedrooms",
                           min_value=1,
                           max_value=6,
                           value=3)

age = st.sidebar.slider("Age of House (years)",
                       min_value=0,
                       max_value=50,
                       value=10)

# Main content area with tabs
tab1, tab2, tab3 = st.tabs(["Prediction", "Data Analysis", "Model Performance"])

with tab1:
    st.header("Price Prediction")
    
    # Create prediction button
    if st.button("Predict Price"):
        # Make prediction
        new_house = pd.DataFrame({
            'Area': [area],
            'Bedrooms': [bedrooms],
            'Age': [age]
        })
        prediction = model.predict(new_house)[0]
        
        # Display prediction with formatting
        st.markdown("### Predicted House Price")
        st.markdown(f"<h2 style='color: #1f77b4;'>${prediction:,.2f}</h2>", unsafe_allow_html=True)
        
        # Display input summary
        st.markdown("### Input Summary")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Area", f"{area} sq ft")
        with col2:
            st.metric("Bedrooms", bedrooms)
        with col3:
            st.metric("Age", f"{age} years")

with tab2:
    st.header("Data Analysis")
    
    # Show correlation heatmap
    st.subheader("Correlation Heatmap")
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(df.corr(), annot=True, cmap='coolwarm', center=0)
    st.pyplot(fig)
    
    # Show scatter plot
    st.subheader("House Price vs Area")
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.scatterplot(data=df, x='Area', y='Price')
    plt.title('House Price vs Area')
    st.pyplot(fig)
    
    # Show basic statistics
    st.subheader("Dataset Statistics")
    st.dataframe(df.describe())

with tab3:
    st.header("Model Performance")
    
    # Calculate and display metrics
    y_pred = model.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("R² Score", f"{r2:.4f}")
    with col2:
        st.metric("Mean Squared Error", f"{mse:,.2f}")
    with col3:
        st.metric("Root Mean Squared Error", f"${rmse:,.2f}")
    
    # Show model coefficients
    st.subheader("Model Coefficients")
    coef_df = pd.DataFrame({
        'Feature': ['Area', 'Bedrooms', 'Age'],
        'Coefficient': model.coef_
    })
    st.dataframe(coef_df)
    
    st.markdown("""
    ### Interpretation:
    - **R² Score**: Shows how well the model fits the data (1.0 is perfect)
    - **RMSE**: Average prediction error in dollars
    - **Coefficients**: Show how much each feature affects the price
    """)

# Footer
st.markdown("---")
st.markdown("Built with Streamlit • Data Science Project") 