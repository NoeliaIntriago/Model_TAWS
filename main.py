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
async def get_prediction(longitud_sepalo: float=0, 
                            ancho_sepalo: float=0, 
                            longitud_petalo: float=0, 
                            ancho_petalo: float=0):
    data = np.array([[longitud_sepalo, ancho_sepalo, longitud_petalo, ancho_petalo]])
    if (longitud_sepalo < 0) or (ancho_sepalo < 0) or (longitud_petalo < 0) or (ancho_petalo < 0):
        raise HTTPException(status_code=422, detail="Datos erróneos")
    else:
        prediction = clf.model.predict(data).tolist()[0]
        flores = {0:"Setosa", 1:"Versicolor", 2:"Virginica"}
        probability = clf.model.predict_proba(data).tolist()
        return {"prediction": flores[prediction],
                "probability": probability}
