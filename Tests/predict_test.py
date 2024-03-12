import requests
import pandas as pd
import json
from pprint import pprint as pp
from sklearn.metrics import classification_report
import numpy as np
from sklearn.utils import shuffle
import random as rnd
#predict endpoint
PREDICT_ENDPOINT = 'http://<ip>:<port>/predict'
SLICE = 15000



def main():
    X_test = pd.read_csv('../data/test_sample2.csv')
    y_test = pd.read_csv('../data/test_sample2_target.csv')
    
    X_test_shuffled, y_test_shuffled = shuffle(X_test, y_test, random_state=rnd.randint(0,len(y_test)))
    flow_request = {
        "model": "RandomForest",
        "query": X_test_shuffled.iloc[:SLICE].to_dict(orient = 'records')
    }

    # pp(json.dumps(flow_request))
    
    response = requests.post(PREDICT_ENDPOINT, json=json.dumps(flow_request),
                             headers = {"Content-Type": "application/json"})  # Use json parameter instead of data
    
    if response.status_code == 200:
        response_dict = json.loads(response.text)
        target_categories = set(y_test_shuffled.values[:SLICE].flatten())
        class_report = classification_report(response_dict["Predictions"], y_test_shuffled.iloc[:SLICE], target_names=target_categories, digits=6)
        print("Classification Report:\n", class_report)

    else:
        print(f"Error occured!")
        print(response.text)
   
    


if __name__ == '__main__':
    main()