# House Price Prediction Project 🏠

A machine learning project that predicts house prices using features like area, number of bedrooms, and age of the house. Built with Python, Streamlit, and scikit-learn.

## Features

- Interactive web interface using Streamlit
- Real-time price predictions
- Data visualization and analysis
- Model performance metrics
- Easy-to-use sliders for input

## Technologies Used

- Python 3.x
- Streamlit
- pandas
- numpy
- scikit-learn
- matplotlib
- seaborn

## Project Structure

```
house_price_prediction/
├── app.py                 # Streamlit web application
├── house_prediction.py    # Main prediction script
├── house_data.csv        # Dataset
├── requirements.txt      # Project dependencies
└── README.md            # Documentation
```

## Installation

1. Clone the repository:
```bash
git clone https://github.com/muhammadali1473/house_price_prediction.git
cd house_price_prediction
```

2. Create a virtual environment:
```bash
python -m venv venv
```

3. Activate the virtual environment:
- Windows:
```bash
venv\Scripts\activate
```
- Linux/Mac:
```bash
source venv/bin/activate
```

4. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

1. Run the Streamlit app:
```bash
streamlit run app.py
```

2. Open your browser and go to `http://localhost:8501`

3. Use the sliders in the sidebar to input house details:
   - Area (square feet)
   - Number of bedrooms
   - Age of the house

4. Click "Predict Price" to get the estimated house price

## Features Explanation

1. **Price Prediction**
   - Input house details using sliders
   - Get instant price predictions
   - View prediction confidence metrics

2. **Data Analysis**
   - Correlation heatmap
   - Price vs Area scatter plot
   - Statistical summaries

3. **Model Performance**
   - R² Score
   - Mean Squared Error
   - Root Mean Squared Error
   - Feature coefficients

## Model Details

The project uses Linear Regression to predict house prices based on:
- House area (square feet)
- Number of bedrooms
- Age of the house

The model is trained on a dataset of house prices with these features and achieves good prediction accuracy.

## Contributing

Feel free to fork the project and submit pull requests with improvements!

## License

This project is open source and available under the MIT License.

## Contact

Created by [@muhammadali1473](https://github.com/muhammadali1473) 