# Model_TAWS
Despliegue de modelo Machine Learning usando Python + FastAPI + Docker

## Descripción
Para entrenar este modelo de clasificación, se usará el dataset Iris, el cual contiene las siguientes características:
 
 - Longitud del sépalo en cm
 - Ancho del sépalo en cm
 - Longitud del pétalo en cm
 - Ancho del pétalo en cm
 
 Las observaciones podrán ser clasificadas en una de las siguientes 3 clases: Iris Versicolor, Iris Setosa e Iris Virginica.

## Estructura de proyecto
```
model_taws/
|
|___ models/
|     |___ ml/
|     |    |___ classifier.py
|     |    |___ train.py
|     |    |___ iris_dt_vl.joblib
|     |___ Iris.py
|
|___ main.py
|___ Dockerfile
|___ README.md
|___ requirements.txt
|___ .gitignore
```


## 1. Instalar paquete para crear entorno virtual
```
$ pip install virtualenv
```

## 2. Crear un entorno virtual
```
$ virtualenv venv
```

## 3. Activar entorno virtual
```
$ <<ruta>>/venv/Scripts/activate
```

## 4. Instalar paquetes
```
$ pip install -r requirements.txt
```

## 5. Ejecutar aplicación web
```
$ uvicorn main:app --reload
```
## 6. Dockerizar aplicación web
```
$ docker build . -t iris_ml_docker
$ docker run -p 8000:8000 iris_ml_docker
```