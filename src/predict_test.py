import requests
import pandas as pd
import json
from pprint import pprint as pp

PREDICT_ENDPOINT = 'http://localhost:8891/predict'

# class CustomEncoder(json.JSONEncoder):
#     def default(self, obj):
#         if np.isinf(obj) or np.isnan(obj) or pd.isnan(obj):
#             return None
#         return json.JSONEncoder.default(self, obj)

def main():
    test_set = pd.read_csv('/home/dimet/Documents/Dimitris/ceid/5G_Intrusion_detection/data/test_sample2.csv')
    y_test = pd.read_csv('/home/dimet/Documents/Dimitris/ceid/5G_Intrusion_detection/data/test_sample2_target.csv')
    # test_set.replace(to_replace = pd.NA, value = None, inplace = True)
    flow_request = {
        "model": "RandomForest",
        "query": test_set.iloc[:1000,:].to_dict(orient = 'records')
    }
    # pp(flow_request)
    # print(type(flow_request['query'][1]['Flgs']))
    pp(json.dumps(flow_request))
    
    response = requests.post(PREDICT_ENDPOINT, json=json.dumps(flow_request),
                             headers = {"Content-Type": "application/json"})  # Use json parameter instead of data
    
    if response.status_code == 200:
        print(f"{'='*10} Response {'='*10}")
        boolar = response["Predictions"] == y_test
        print(f"{'='*10} Response {'='*10}")
        print(f"{type(boolar)}\n{len(boolar)}")
    else:
        print(response.text)
   
    


if __name__ == '__main__':
    main()