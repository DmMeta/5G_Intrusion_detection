import requests
import pandas as pd
import json
from sklearn.metrics import classification_report
from sklearn.utils import shuffle
import random as rnd

# Predict endpoint
PREDICT_ENDPOINT = 'http://127.0.0.1:8891/predict'
SLICE = 15000  # Number of samples to use

def main():
    # Read test data
    X_test = pd.read_csv('../data/test_sample.csv')
    y_test = pd.read_csv('../data/test_sample_labels.csv')
    
    # Shuffle test data
    X_test_shuffled, y_test_shuffled = shuffle(X_test, y_test, random_state=rnd.randint(0,len(y_test)))
    
    # Prepare request data
    flow_request = {
        "model": "CNN_LSTM",  # Specify the model
        "query": X_test_shuffled.iloc[:SLICE].to_dict(orient='records')  # Convert test data to dictionary
    }

    # Send POST request to the predict endpoint
    response = requests.post(PREDICT_ENDPOINT, json=json.dumps(flow_request),
                             headers={"Content-Type": "application/json"})  
    
    # Check response status
    if response.status_code == 200:
        # If successful, parse response and generate classification report
        response_dict = json.loads(response.text)
        target_categories = set(y_test_shuffled.values[:SLICE].flatten())
        class_report = classification_report(response_dict["Predictions"], y_test_shuffled.iloc[:SLICE], target_names=target_categories, digits=6)
        print("Classification Report:\n", class_report)
    else:
        # If error occurs, print error message
        print(f"Error occurred!")
        print(response.text)
   
if __name__ == '__main__':
    main()  # Call the main function
