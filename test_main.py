from fastapi.testclient import TestClient
import models.ml.classifier as clf
from main import app


client = TestClient(app)


def test_read_main():
    with TestClient(app) as client:
        response = client.post("/predict",params={"longitud_sepalo":0,
                                                "ancho_sepalo":0,
                                                "longitud_petalo":0,
                                                "ancho_petalo":0})
        assert response.status_code == 200
        assert response.json() == {"prediction": "Setosa",
                                    "probability": [[1.0, 0.0, 0.0]]}
        
    
