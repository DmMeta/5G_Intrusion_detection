from fastapi import (FastAPI,
                     HTTPException,
                     Request,
                     Response,
                     status) 
import json
from model import Modelselector 

app = FastAPI()

@app.post("/predict")
async def predict(pred_request: Request):
    
    payload = await pred_request.json()
    pred_request = json.loads(payload)
    model_name = pred_request['model']
    y = pred_request['query']
    print(f"Model: {model_name} Query: {len(y)}")
    
    model = Modelselector(model_name)
    
    prediction = model.predict(y)
    
    
    return prediction

@app.get("/health")
def health():
    return {"status": "ok"}