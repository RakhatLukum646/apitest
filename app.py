import re
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel
import uvicorn
import pickle
import pandas as pd
import os
app = FastAPI()

with open('model.pkl', 'rb') as f:
    model = pickle.load(f)

request_counter = 0

class PredictionRequest(BaseModel):
    Pclass: int
    Age: int
    Fare: float

@app.get('/')
async def stats():
    return {
        'request_counter': request_counter
    }

@app.get("/health")
async def heatlh():
    return {"status": "OK"}

@app.post("/predict")
async def predict(input_data: PredictionRequest):
    global request_counter
    request_counter += 1

    new_data = pd.DataFrame({
    'Pclass': [input_data.Pclass],
    'Age': [input_data.Age],
    'Fare': [input_data.Fare]
    })

    prediction = model.predict(new_data)

    result = "Not Survived" if prediction[0] == 0 else "Survived"

    return {"prediction": result}

if __name__ == '__main__':
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", 8080)))
