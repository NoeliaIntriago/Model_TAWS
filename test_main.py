from fastapi.testclient import TestClient
import main

client = TestClient(main.app)

def test_predict():
    response = client.post("/predict", params={"longitud_sepalo":4.8, 
                                                "ancho_sepalo":1.3, 
                                                "longitud_petalo": 1, 
                                                "ancho_petalo": 2})
    assert response.status_code == 200