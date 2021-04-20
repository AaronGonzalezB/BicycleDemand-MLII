import uvicorn
from fastapi import FastAPI
from model.model import BicycleDemandParams, BicycleDemandModel

# 2. Creamos la app y la estructura del modelo
app = FastAPI()
model = BicycleDemandModel()

# 3. Exponemos la funcionalidad de la prediccion, hacemos una prediccion a partir del JSON suministrado
@app.post('/predict')
def predict_demand(bicycle: BicycleDemandParams):
    data = bicycle.dict()
    prediction = model.predict_demand(
                    #data['yr'],
                    data['mnth'],
                    data['hr'],
                    data['holiday'],
                    data['weekday'],
                    data['workingday'],
                    data['weathersit'],
                    data['temp'],
                    data['hum'],
                    data['windspeed']
    )

    return {
        'prediction':prediction,
    }

# 4. Ejecucion de la API
if __name__ == '__main__':
    uvicorn.run(app, host='127.0.0.1', port=8000)