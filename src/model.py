import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
import os
# import _pickle as pickle
import pandas as pd
 
# class ModelUntrainedException(Exception):
#     def __init__(self, message = "The model parameters weren't loaded so the model remains untrained!"):
#         self.message = message
#         super().__init__(self.message)


class Modelselector():
    
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
        
        import yaml
        model = None

        with open('config.yaml', 'r') as file:
            config = yaml.safe_load(file)
            
        if self.model_name == 'ANN':
            ann_config = config['ANN']
            from package_src.analysis_training.neural_networks.model_definition import ANN_BinaryClassifier
            model = ANN_BinaryClassifier(input_size = ann_config['input_size'], hidden_size = ann_config['hidden_size'], 
                                         output_size = ann_config['output_size'])
            
            filePath = os.path.join("../models","ANN")
            with open(filePath, "rb") as fp:
                model.load_state_dict(joblib.load(fp))

        elif self.model_name == 'CNN':
            cnn_config = config['CNN']
            from package_src.analysis_training.neural_networks.model_definition import CNN_BinaryClassifier
            model = CNN_BinaryClassifier(input_channels = cnn_config['input_channels'], kernels = cnn_config['kernels'], 
                                         kernel_size = cnn_config['kernel_size'], classes = cnn_config['classes'])
            
            filePath = os.path.join("../models","CNN")
            with open(filePath, "rb") as fp:
                model.load_state_dict(joblib.load(fp))
        
        elif self.model_name == 'CNN_LSTM':
            cnn_lstm_config = config['CNN_LSTM']
            from package_src.analysis_training.neural_networks.model_definition import CNN_LSTM_BinaryClassifier
            model = CNN_LSTM_BinaryClassifier(input_channels = cnn_lstm_config['input_channels'], kernels = cnn_lstm_config['kernels'], 
                                            kernel_size = cnn_lstm_config['kernel_size'], lstm_hidden_size = cnn_lstm_config['lstm_hidden_size'], 
                                            lstm_num_layers = cnn_lstm_config['lstm_num_layers'], classes = cnn_lstm_config['classes'])
            
            filePath = os.path.join("../models","CNN_LSTM")
            with open(filePath, "rb") as fp:
                model.load_state_dict(joblib.load(fp))
        elif self.model_name in ['RandomForest', 'DecisionTree']:
            filePath = os.path.join("../models",f"{self.model_name}.pkl")
            model = joblib.load(filePath)
        
        else:
            print(f"Model {self.model_name} not found! Loading untrained model") 
            self.model_name = "DecisionTree"  
            model = DecisionTreeClassifier()
       
        return model
    
    
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
        # y_out = pd.DataFrame(columns = self.features)
        # for query_flow in data:
        #     q_flow = {key:[val] for key, val in query_flow.items()}
        #     # features_index = [f"feature {i}" for i,_ in enumerate(query_flow)]

        #     flow_df = pd.DataFrame(q_flow)
        #     y_out = pd.concat([y_out, flow_df[self.features]], axis = 0)
        
        dfs = []

        for query_flow in data:
            q_flow = {key: [val] for key, val in query_flow.items()}
            flow_df = pd.DataFrame(q_flow)
            if not flow_df.empty:  
                dfs.append(flow_df[self.features])

        y_out = pd.concat(dfs, axis=0, ignore_index=False)
        
        if self.model_name in ['ANN', 'CNN', 'CNN_LSTM']:
            filePath = os.path.join("../models", "Preprocessors.pkl")
            with open(filePath, 'rb') as file:
            
                preprocessor = joblib.load(file)

            pipeline = Pipeline([
                    ('preprocessor', preprocessor),
                ])
            y_out = pipeline.transform(y_out)
        
        return y_out
                    


    
    def predict(self, data):
        
        y = self._preprocess(data)
        
       
        if self.model_name not in ['RandomForest', 'DecisionTree']:
            outputs = self.model.model(y)
            enc_predictions = (outputs > 0.5).float()
            try: 
                filePath = os.path.join("../models","LabelEncoder.pkl")
                with open(filePath, "rb") as file:
                    encoder = joblib.load(file)
            except FileNotFoundError as e:
                print(f"{e} \nLabel Encoder Not found!")
            
            predictions = encoder.inverse_transform(enc_predictions)
            print(predictions)
            return {"Predictions": predictions.tolist()}
        
        predictions = self.model.predict(y)
            
        return {"Predictions": predictions.tolist()}
    
            
   