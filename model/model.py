import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin, RegressorMixin
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
from pydantic import BaseModel
import pickle

# 2. Clase base, se incluyen solo los parámetros necesarios par ala predicciónÑ
class BicycleDemandParams(BaseModel):
    mnth: int
    hr: int
    holiday: int
    weekday: int
    workingday: int
    weathersit: float
    temp: float
    hum: float
    windspeed: float

# 3. Clase para entrenar el modelo y hacer las predicciones
class BicycleDemandModel:
    # Carga el dataset y el modelo si existe, si no, llama el metodo de train model y guarda el modelo
    def __init__(self):
        self.df = pd.read_csv('data/timeseries.csv')
        self.model_fname_ = 'RFregressor.pkl'
        try:
            self.model = pickle.load(open('model/RFregressor.pkl', 'rb'))
        except Exception as _:
            self.model = self._train_model()
            pickle.dump(rf_rg, open('models/RFregressor.pkl', 'wb'))

    # 7. Ejecución del entrenamiento bajo el modelo in-house
    def _train_model_(self):

        params= pickle.load(open('local/RFparams.pkl', 'rb'))
        max_depth, n_estimators = list(params.values())

        model_def = Pipeline([
        
            ("extractor", BikeRentalFeatureExtractor()),
            ("important", SelectImportantFeatures()),
            ("selector", BikeRentalFeatureSelector()),
            (("scaler", MinMaxScaler())),
            ("regressor", RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth))

            ])
        model_def.fit(df)

        return model_def    

    # 8. Prediccion a partir de datos ingresados por el usuario
    def predict_demand(self,mnth,hr,holiday,weekday,workingday,weathersit,temp,hum,windspeed):
            data_in = [[mnth,hr,holiday,weekday,workingday,weathersit,temp,hum,windspeed]]
            prediction = self.model.predict(data_in)
            return prediction[0]

# 4. Clase para extraccion de caracteristicas
class BikeRentalFeatureExtractor(BaseEstimator, TransformerMixin):
  
  def fit(self, X, y=None):
    return self

  def transform(self, X):
    all_indices = X.index
    null_indices = X.loc[X.isnull().any(axis=1)].index
    deltahour = datetime.timedelta(hours=1)
    deltaday = datetime.timedelta(days=1)

    cols_to_copy = [
        'season', 'holiday', 'weekday', 'workingday' , 'weathersit' , 'temp',
        'atemp', 'hum', 'windspeed' , 'casual', 'registered', 'cnt']

    for index in null_indices:       
        
        X.loc[index, ['hr', 'yr', 'mnth']] = (
            index.hour, int(index.year == 2012), index.month)
        
        if (index - deltahour).dayofweek == index.dayofweek:
            similar_row = X.loc[index - deltahour]
        else:
            similar_row = X.loc[index + deltahour]
            
        if np.sum(similar_row.isnull()) != 0:
            
            similar_row = X.loc[index - deltaday]
            
        X.loc[index, cols_to_copy] = similar_row[cols_to_copy]
        
    second_week_start = X.index[0] + datetime.timedelta(days=6, hours=23)
    last_week_start = X.index[-1] - datetime.timedelta(weeks=1)
    values = np.concatenate( (X.loc[:second_week_start,'cnt'].values, X.loc[:last_week_start,'cnt'].values) )
    X['last_week_same_hour'] = values

    second_day_start = X.index[0] + datetime.timedelta(hours=23)
    last_day_start = X.index[-1] - datetime.timedelta(days=1)
    values = np.concatenate( (X.loc[:second_day_start,'cnt'].values,X.loc[:last_day_start,'cnt'].values))
    X['last_day_same_hour'] = values

    return X

# 5. Clase para identificar las caracteristicas importantes
class SelectImportantFeatures(BaseEstimator, TransformerMixin):

  def fit(self, X, y=None):
    return self
  
  def transform(self,X):

    predictor_cols = ["mnth","hr","holiday","weekday","workingday","weathersit","temp","hum","windspeed"]
    corr_features = set()
    corr_matrix = X[predictor_cols].corr()

    for i in range(len(corr_matrix.columns)):
      for j in range(i):
          if abs(corr_matrix.iloc[i, j]) > 0.8:
              colname = corr_matrix.columns[i]
              corr_features.add(colname)

    X.drop(labels=corr_features, axis=1, inplace=True)

    return X

# 6. Clase para seleccionar los features para entrenamiento y validacion
class BikeRentalFeatureSelector(BaseEstimator, TransformerMixin):
  def __init__(self):
    self.feature_columns = feature_columns

  def fit(self, X, y=None):

    return self
  
  def transform(self,X):

    self.feature_columns = ["yr", "mnth","hr","holiday","weekday","workingday","weathersit","temp","hum","windspeed"]

    train_indices = X["yr"] == 0
    test_indices = ~train_indices

    X_train, y_train = X[self.feature_columns][train_indices], y[train_indices]
    X_test, y_test = X[self.feature_columns][test_indices], y[test_indices]
    X_train.drop('yr', inplace=True, axis=1)
    X_test.drop('yr', inplace=True, axis=1)

    return X_train, X_test, y_train, y_test


