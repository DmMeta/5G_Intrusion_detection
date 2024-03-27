from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel
import json
from model import ModelAdapter  # Importing ModelAdapter from the model module
from prometheus_fastapi_instrumentator import Instrumentator, metrics

app = FastAPI()  # Creating a FastAPI application instance
Instrumentator().instrument(app).expose(app)

class PredictionQuery(BaseModel):
    model: str
    query: list

    class Config:
        arbitrary_types_allowed = True

@app.post("/predict")  # Endpoint for making predictions
async def predict(pred_request: Request):
    # Get JSON payload from the request
    payload = await pred_request.json()
    pred_request = json.loads(payload)

    # Extract model name and query data from the payload
    model_name = pred_request['model']
    y = pred_request['query']
    print(f"Model: {model_name} | Number of Samples: {len(y)}")
    
    try:
        # Select the appropriate model based on the model name
        model = ModelAdapter.select_model(model_name)
        # Make predictions using the selected model
        prediction = model.predict(y)
    except ValueError as e:
        # Handle errors if the model is not found
        raise HTTPException(status_code=404, detail=str(e))

    # Return the prediction
    return prediction

@app.get("/health")  # Endpoint for checking the health of the application
def health():
    return {"status": "ok"}  # Return a JSON response indicating the health status
