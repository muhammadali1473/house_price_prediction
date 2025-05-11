import requests
import json

# Test data
data = {
    "area": 1500,
    "bedrooms": 3,
    "age": 5
}

# Make the prediction request
response = requests.post("http://127.0.0.1:8000/predict", json=data)

# Print the response
print("Status Code:", response.status_code)
print("Response:", response.json()) 