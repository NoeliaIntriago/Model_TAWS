from fastapi.testclient import TestClient
import models.ml.classifier as clf
from main import app


client = TestClient(app)


def test_read_main():
    with TestClient(app) as client:
        response = client.post("/predict",params={"param1":0,
                                                "param2":0,
                                                "param3":0,
                                                "param4":0})
        assert response.status_code == 200
        assert response.json() == {"prediction": "Setosa",
                                    "probability": [[1.0, 0.0, 0.0]]}
        
    
