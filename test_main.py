from fastapi.testclient import TestClient
import models.ml.classifier as clf
from main import app


client = TestClient(app)


def test_predict_status():
    with TestClient(app) as client:
        response = client.post("/predict",params={"param1":1,
                                                "param2":2,
                                                "param3":2,
                                                "param4":2})
        assert response.status_code == 200
        assert response.json() == {"prediction": "Setosa",
                                    "probability": [[1.0, 0.0, 0.0]]}

def test_when_params_le_zero():
    with TestClient(app) as client:
        response = client.post("/predict",params={"param1":-1,
                                                "param2":1,
                                                "param3":1,
                                                "param4":1})
        assert response.status_code == 422
        assert response.json() =={'detail': [{'loc': ['query', 'param1'],
                                               'msg': 'ensure this value is greater than 0', 
                                               'type': 'value_error.number.not_gt', 
                                               'ctx': {'limit_value': 0}}]}


def test_when_params_is_str():
    with TestClient(app) as client:
        response = client.post("/predict",params={"param1":"a",
                                                    "param2":1,
                                                    "param3":1,
                                                    "param4":1})
        assert response.status_code == 422
        assert response.json() == {'detail': [{'loc': ['query', 'param1'], 
                                               'msg': 'value is not a valid float', 'type': 'type_error.float'}]} != {'detail': [{'ctx': {}, 'loc': ['query', 'param1'], 
                                               'msg': 'value is not a valid float', 'type': 'type_error.float'}]}
        
        

        


        
    
