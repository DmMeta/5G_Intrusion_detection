from abc import ABC, abstractmethod
import yaml
import joblib
import torch
import pandas as pd

class ModelAdapter(ABC):

    def __init__(self, config_path="./config.yml"):
        self._config = self._load_config(config_path)
        self.features = self._load_features(self._config['features_path'])

    @abstractmethod
    def _load_model(self,):
        pass
    @abstractmethod
    def _preprocess(self,):
        pass
    @abstractmethod
    def predict(self,):
        pass
    
    def _load_config(self, config_path):
        with open(config_path, 'r') as stream:
            try:
                return yaml.safe_load(stream)
            except yaml.YAMLError as exc:
                print(exc)
                return None

    def _load_features(self, features_path):
        with open(features_path, 'rb') as file:
            features = joblib.load(file)
        return features


    @classmethod
    def select_model(cls, model_name = "RandomForest"):
        if model_name == 'ANN':
            return ANNAdapter()
        elif model_name == "CNN":
            return CNNAdapter()
        elif model_name == "CNN_LSTM":
            return CNN_LSTMAdapter()
        elif model_name in ['RandomForest', 'DecisionTree']:
            return ScikitTreeBasedModel(model_name)
        else:
            raise ValueError("Invalid model type. Not supported")
# Slight modification to the way the model is selected in server
# now through ModelAdapter.select_model(model_name)


class ANNAdapter(ModelAdapter):
    def __init__(self):
        super(ANNAdapter, self).__init__()
        self.model = self._load_model()

    def _load_model(self,):
        from model_definition import ANN_BinaryClassifier

        input_size = self._config['NN']['ANN']['input_size']
        hidden_size = self._config['NN']['ANN']['hidden_size']
        output_size = self._config['NN']['ANN']['output_size']

        model = ANN_BinaryClassifier(input_size = input_size, 
                                     hidden_size = hidden_size, 
                                     output_size = output_size)
        with open(self._config['NN']['ANN']['model_path'], 'rb') as file:
            model.load_state_dict(joblib.load(file))

        return model

    def _preprocess(self, data):
        dfs = []

        for query_flow in data:
            q_flow = {key: [val] for key, val in query_flow.items()}
            flow_df = pd.DataFrame(q_flow)
            if not flow_df.empty:  
                dfs.append(flow_df[self.features])

        y_out = pd.concat(dfs, axis=0, ignore_index=False)

        ann_preprocessor_path = self._config['NN']['ANN']['preprocessor_path']
        with open(ann_preprocessor_path, 'rb') as file:  
            preprocessor = joblib.load(file)

        y_out = preprocessor.transform(y_out)

        return torch.FloatTensor(y_out)

    def predict(self, data):
        y = self._preprocess(data)
        y_pred = (self.model(y) > 0.5).int()

        target_encoder_path = self._config['NN']['target_encoder_path']
        with open(target_encoder_path, "rb") as file:
            t_encoder = joblib.load(file)
        
        y_pred_labels = t_encoder.inverse_transform(y_pred.ravel())
        return {"Predictions": y_pred_labels.tolist()}


class CNNAdapter(ModelAdapter):
    def __init__(self):
        super(CNNAdapter, self).__init__()
        self.model = self._load_model()

    def _load_model(self,):
        from model_definition import CNN_BinaryClassifier

        classes = self._config['NN']['CNN']['classes']
        input_channels = self._config['NN']['CNN']['input_channels']
        input_features = self._config['NN']['CNN']['input_features']
        kernels = self._config['NN']['CNN']['kernels']
        kernel_size = self._config['NN']['CNN']['kernel_size']

        model = CNN_BinaryClassifier(input_channels = input_channels, 
                                     input_features = input_features, 
                                     kernels = kernels, kernel_size=kernel_size, classes=classes)
        with open(self._config['NN']['CNN']['model_path'], 'rb') as file:
            model.load_state_dict(joblib.load(file))

        return model

    def _preprocess(self, data):
        dfs = []

        for query_flow in data:
            q_flow = {key: [val] for key, val in query_flow.items()}
            flow_df = pd.DataFrame(q_flow)
            if not flow_df.empty:  
                dfs.append(flow_df[self.features])

        y_out = pd.concat(dfs, axis=0, ignore_index=False)

        cnn_preprocessor_path = self._config['NN']['CNN']['preprocessor_path']
        with open(cnn_preprocessor_path, 'rb') as file:  
            preprocessor = joblib.load(file)

        last_dim = int(self._config['NN']['CNN']['input_features'])
        y_out = preprocessor.transform(y_out).reshape(-1, 1, last_dim )

        return torch.FloatTensor(y_out)

    def predict(self, data):
        y = self._preprocess(data)

        y_pred = (self.model(y) > 0.5).int()

        target_encoder_path = self._config['NN']['target_encoder_path']
        with open(target_encoder_path, "rb") as file:
            t_encoder = joblib.load(file)
        
        y_pred_labels = t_encoder.inverse_transform(y_pred.ravel())
        return {"Predictions": y_pred_labels.tolist()}



class CNN_LSTMAdapter(ModelAdapter):
    def __init__(self):
        super(CNN_LSTMAdapter, self).__init__()
        self.model = self._load_model()

    def _load_model(self,):
        from model_definition import CNN_LSTM_BinaryClassifier

        classes = self._config['NN']['CNN_LSTM']['classes']
        input_channels = self._config['NN']['CNN_LSTM']['input_channels']
        kernels = self._config['NN']['CNN_LSTM']['kernels']
        kernel_size = self._config['NN']['CNN_LSTM']['kernel_size']
        lstm_num_layers = self._config['NN']['CNN_LSTM']['lstm_num_layers']
        lstm_hidden_size = self._config['NN']['CNN_LSTM']['lstm_hidden_size']

        model = CNN_LSTM_BinaryClassifier(input_channels = input_channels, 
                                          kernels = kernels, 
                                          kernel_size=kernel_size, 
                                          classes=classes,
                                          lstm_hidden_size=lstm_hidden_size, lstm_num_layers=lstm_num_layers
                                          )
        with open(self._config['NN']['CNN_LSTM']['model_path'], 'rb') as file:
            model.load_state_dict(joblib.load(file))

        return model

    def _preprocess(self, data):
        dfs = []

        for query_flow in data:
            q_flow = {key: [val] for key, val in query_flow.items()}
            flow_df = pd.DataFrame(q_flow)
            if not flow_df.empty:  
                dfs.append(flow_df[self.features])

        y_out = pd.concat(dfs, axis=0, ignore_index=False)

        cnn_lstm_preprocessor_path = self._config['NN']['CNN_LSTM']['preprocessor_path']
        with open(cnn_lstm_preprocessor_path, 'rb') as file:  
            preprocessor = joblib.load(file)

        last_dim = len(self.features)
        y_out = preprocessor.transform(y_out).reshape(-1, 1, last_dim )

        return torch.FloatTensor(y_out)

    def predict(self, data):
        y = self._preprocess(data)
        y_pred = (self.model(y) > 0.5).int()

        target_encoder_path = self._config['NN']['target_encoder_path']
        with open(target_encoder_path, "rb") as file:
            t_encoder = joblib.load(file)
        
        y_pred_labels = t_encoder.inverse_transform(y_pred.ravel())
        return {"Predictions": y_pred_labels.tolist()}


class ScikitTreeBasedModel(ModelAdapter):
    def __init__(self, model_name):
        super(ScikitTreeBasedModel, self).__init__()
        self.model_type = model_name
        self.model = self._load_model()

    def _load_model(self,):
        model_path = self._config['TREE_BASED_MODELS'][self.model_type]['model_path']
        with open(model_path, 'rb') as file:
            model = joblib.load(file)
        return model

    def _preprocess(self, data):
        dfs = []

        for query_flow in data:
            q_flow = {key: [val] for key, val in query_flow.items()}
            flow_df = pd.DataFrame(q_flow)
            if not flow_df.empty:  
                dfs.append(flow_df[self.features])

        y_out = pd.concat(dfs, axis=0, ignore_index=False)

        return y_out
    
    def predict(self, data):
        y = self._preprocess(data)

        y_pred = self.model.predict(y)
       
        return {"Predictions": y_pred.tolist()}
