from fastapi import (FastAPI,
                     HTTPException,
                     Request,
                     Response,
                     status) 
from pydantic import BaseModel
from typing import Dict, List, Union, Any
import json
from model import Modelselector 

# with open(os.path.join("../models","top_10_features_cols.pkl"), "rb") as ffile:
#     features = pickle.load(ffile)

# field_definitions = {field: Union[float,str] for field in features}

# Query = create_model("Query", **field_definitions)

# print(Query.__annotations__.keys())

app = FastAPI()

class PredictionQuery(BaseModel):
    model: str
    query: list

    class Config:
        arbitrary_types_allowed = True

'''

{
    model: "RandomForest",
    query: [
        {
            "age": 25,
        },
        {
            "age": 30,
        }
}


'''


@app.post("/predict")
async def predict(pred_request: Request):
    
    payload = await pred_request.json()
    pred_request = json.loads(payload)
    model_name = pred_request['model']
    y = pred_request['query']
    print(f"Model: {model_name} Query: {len(y)}")
    
    model = Modelselector(model_name)
    
    prediction = model.predict(y)
    
    # prediction_resp = {"Prediction": prediction[0]}
    return prediction

@app.get("/health")
def health():
    return {"status": "ok"}