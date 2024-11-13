# app.py
from fastapi import FastAPI
import tensorflow as tf
import numpy as np
from pydantic import BaseModel

app = FastAPI()

# Load your model
model = tf.keras.models.load_model("model.h5")

# Define input data structure
class ModelInput(BaseModel):
    data: list

@app.post("/predict")
async def predict(input_data: ModelInput):
    input_array = np.array([input_data.data])
    prediction = model.predict(input_array)
    return {"prediction": prediction.tolist()}
