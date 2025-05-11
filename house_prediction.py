import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error

# Set the style for better visualizations
plt.style.use('default')
sns.set_theme()

def load_and_explore_data():
    # Load the dataset
    df = pd.read_csv('house_data.csv')
    print("\n=== Dataset First 5 Rows ===")
    print(df.head())
    
    print("\n=== Dataset Information ===")
    print(df.info())
    
    print("\n=== Statistical Summary ===")
    print(df.describe())
    
    print("\n=== Missing Values ===")
    print(df.isnull().sum())
    return df

def visualize_data(df):
    # Create a correlation heatmap
    plt.figure(figsize=(10, 8))
    sns.heatmap(df.corr(), annot=True, cmap='coolwarm', center=0)
    plt.title('Correlation Heatmap')
    plt.show()

    # Relationship between Area and Price
    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=df, x='Area', y='Price')
    plt.title('House Price vs Area')
    plt.show()

def train_model(df):
    # Prepare data for model training
    X = df[['Area', 'Bedrooms', 'Age']]
    y = df['Price']

    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Create and train the model
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Print model coefficients
    print("\n=== Model Coefficients ===")
    for feature, coef in zip(X.columns, model.coef_):
        print(f"{feature}: {coef:.2f}")
    print(f"Intercept: {model.intercept_:.2f}")
    
    # Calculate performance metrics
    r2 = r2_score(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    
    print("\n=== Model Performance ===")
    print(f"R² Score: {r2:.4f}")
    print(f"Mean Squared Error: {mse:.2f}")
    print(f"Root Mean Squared Error: {rmse:.2f}")
    
    return model

def predict_new_house(model):
    # Create a sample house
    new_house = pd.DataFrame({
        'Area': [2500],
        'Bedrooms': [3],
        'Age': [10]
    })
    
    # Make prediction
    predicted_price = model.predict(new_house)[0]
    
    print("\n=== New House Prediction ===")
    print("New House Details:")
    print(f"Area: 2500 sq ft")
    print(f"Bedrooms: 3")
    print(f"Age: 10 years")
    print(f"\nPredicted Price: ${predicted_price:,.2f}")

def main():
    print("=== House Price Prediction Model ===")
    
    # Load and explore data
    df = load_and_explore_data()
    
    # Visualize data
    visualize_data(df)
    
    # Train model
    model = train_model(df)
    
    # Make prediction for a new house
    predict_new_house(model)

if __name__ == "__main__":
    main() 