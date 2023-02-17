from fastapi.testclient import TestClient
import models.ml.classifier as clf
from main import app


client = TestClient(app)


def test_predict_status():
    with TestClient(app) as client:
        response = client.post("/predict",params={"longitud_sepalo":1,
                                                "ancho_sepalo":2,
                                                "longitud_petalo":2,
                                                "ancho_petalo":2})
        assert response.status_code == 200
        assert response.json() == {"prediction": "Setosa",
                                    "probability": [[1.0, 0.0, 0.0]]}

def test_when_params_le_zero():
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
        
        

        


        
    
