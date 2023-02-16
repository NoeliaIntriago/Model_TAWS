import models.ml.classifier as clf
import numpy as np
from fastapi import FastAPI, HTTPException
from joblib import load
from models.Iris import Iris

app = FastAPI(title="Iris ML API", 
                description="""API para modelo ML del dataset Iris\n
                Valores de predicción:\n
                0 -> Iris Setosa\n
                1 -> Iris Versicolor\n
                2 -> Iris Virginica""",
                version="1.0")

@app.get('/')
def home():
    return {"Information": "API para modelo ML del dataset Iris",
                "Predicción": "Interpretación",
                "0": "Iris Setosa",
                "1": "Iris Versicolor",
                "2": "Iris Virginica"}

# Importa modelo solo al iniciar, no cada que se realiza una petición
@app.on_event('startup')
async def load_model():
    clf.model = load('models/ml/iris_dt_vl.joblib')

# Petición para predecir
@app.post('/predict', tags=["Prediction - Iris ML Model"])
async def get_prediction(param1: float=0, param2: float=0, param3: float=0, param4: float=0):
    data = np.array([[param1, param2, param3, param4]])
    prediction = clf.model.predict(data).tolist()[0]
    flores = {0:"Setosa", 1:"Virginica", 2:"Versicolor"}
    probability = clf.model.predict_proba(data).tolist()
    return {"prediction": flores[prediction],
            "probability": probability}
