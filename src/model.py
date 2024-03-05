import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
import os
import _pickle as pickle
import pandas as pd
 
class ModelUntrainedException(Exception):
    def __init__(self, message = "The model parameters weren't loaded so the model remains untrained!"):
        self.message = message
        super().__init__(self.message)


class Modelselector():
    
    model_preferences = {
        "RandomForest": RandomForestClassifier,
        "DecisionTree": DecisionTreeClassifier,
    }
    
    def __init__(self, model_name):
        self.model_name = model_name
        self.model = self._select_model()
        
        
        
        try: 
            filePath = os.path.join("../models","top_10_features_cols.pkl")
            with open(filePath, "rb") as ffile:
                self.features = pickle.load(ffile)
        except FileNotFoundError as e:
            print(f"{e} \nCustom features not found, using default features")
            

    def _select_model(self):
        
        try:
            model = Modelselector.model_preferences[self.model_name]()
            filePath = os.path.join("../models",f"{self.model_name}.pkl")
            model = joblib.load(filePath)
            
            return model
            
        except ModelUntrainedException as e:
            print(f"{e} \nUsing default untrained model")
            return Modelselector.model_preferences[self.model_name]()
    
    def _preprocess(self, data):
        # Implement preprocessing here
        '''
        query: [
        {
            "age": 25,
            
        },
        {
            "age": 30,
        }
        '''
        y_out = pd.DataFrame(columns = self.features)
        for query_flow in data:
            q_flow = {key:[val] for key, val in query_flow.items()}
            # features_index = [f"feature {i}" for i,_ in enumerate(query_flow)]
            
            flow_df = pd.DataFrame(q_flow)
            y_out = pd.concat([y_out, flow_df[self.features]], axis = 0)
            

        
        return y_out

    def predict(self, data):
        
        y = self._preprocess(data)
        
        if self.model is not None:
            predictions = self.model.predict(y)
             
            return {"Predictions": predictions.tolist()}
        else :
            raise RuntimeError("Unspecified model. Shouldn't be here...")
            
        