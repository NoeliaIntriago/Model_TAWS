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
    clf.model = load('./models/ml/iris_dt_vl.joblib')

# Petición para predecir
@app.post('/predict', tags=["Prediction - Iris ML Model"])
async def get_prediction(iris: Iris):
    data = dict(iris)['data']
    prediction = clf.model.predict(data).tolist()
    probability = clf.model.predict_proba(data).tolist()
    return {"prediction": prediction,
            "probability": probability}
