from fastapi.testclient import TestClient
import models.ml.classifier as clf
from main import app


client = TestClient(app)


def test_predict_setosa():
    with TestClient(app) as client:
        response = client.post("/predict",params={"longitud_sepalo":5.1,
                                                "ancho_sepalo":3.2,
                                                "longitud_petalo":1.4,
                                                "ancho_petalo":0.2})
        assert response.status_code == 200
        assert response.json() == {"prediction": "Setosa",
                                    "probability": [[1.0, 0.0, 0.0]]}

def test_predict_versicolor():
    with TestClient(app) as client:
        response = client.post("/predict",params={"longitud_sepalo":6.5,
                                                "ancho_sepalo":3.2,
                                                "longitud_petalo":4.6,
                                                "ancho_petalo":1.5})
        assert response.status_code == 200
        assert response.json() == {"prediction": "Versicolor",
                                    "probability": [[0.0, 1.0, 0.0]]}

def test_predict_virginica():
    with TestClient(app) as client:
        response = client.post("/predict",params={"longitud_sepalo":6,
                                                "ancho_sepalo":3.3,
                                                "longitud_petalo":6,
                                                "ancho_petalo":2.5})
        assert response.status_code == 200
        assert response.json() == {"prediction": "Virginica",
                                    "probability": [[0.0, 0.0, 1.0]]}                                                                        

def test_when_params_less_zero():
    with TestClient(app) as client:
        response = client.post("/predict",params={"longitud_sepalo":-1,
                                                "ancho_sepalo":1,
                                                "longitud_petalo":1,
                                                "ancho_petalo":1})
        assert response.status_code == 422
        assert response.json() =={'detail': [{'loc': ['query', 'longitud_sepalo'],
                                               'msg': 'ensure this value is greater than 0', 
                                               'type': 'value_error.number.not_gt', 
                                               'ctx': {'limit_value': 0}}]}


def test_when_params_is_str():
    with TestClient(app) as client:
        response = client.post("/predict",params={"longitud_sepalo":"a",
                                                    "ancho_sepalo":1,
                                                    "longitud_petalo":1,
                                                    "ancho_petalo":1})
        assert response.status_code == 422
        assert response.json() == {'detail': [{'loc': ['query', 'longitud_sepalo'], 
                                               'msg': 'value is not a valid float', 'type': 'type_error.float'}]} != {'detail': [{'ctx': {}, 'loc': ['query', 'longitud_sepalo'], 
                                               'msg': 'value is not a valid float', 'type': 'type_error.float'}]}
        
        

        


        
    
