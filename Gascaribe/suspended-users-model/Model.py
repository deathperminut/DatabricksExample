#%%
import numpy as np
import pandas as pd
import seaborn as sns

from sklearn.feature_selection import f_regression, mutual_info_regression
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler

#%%
class SuspendedUsersModel:
    lab_var = 'TiempoTotalSuspension'

    con_var = [
        'DeudaCorrienteNoVencida',
        'DeudaCorrienteVencida',
        'DeudaDiferida',
        'CantRefiUltimoAño',
        'CantHistoriaRefi',
        'Veces30',
        'Veces60',
        'Veces90',
        'VecesMas90',
        'MoraMaxima',
        'Suspensiones',
        'Reconexiones',
        'Cuota',
        'DiasSuspendido',
    ]

    cat_var = [
        'Estrato',
        'Refinanciado',
    ]

    def __init__(self, n_neighbors = 0):
        self.n_neighbors = n_neighbors

        if self.n_neighbors % 2 != 1 or self.n_neighbors <= 0:
            raise AssertionError('n_neighbors must be an odd, positive integer.')

        self.one_hot_encoder = OneHotEncoder(drop = 'first')
        self.standard_scaler = StandardScaler()
        self.predictor = NearestNeighbors(n_neighbors = self.n_neighbors)
        self.x = None
        self.y = None

    def encode(self, frame):
        return np.append(
            frame[self.__class__.con_var].to_numpy(),
            self.one_hot_encoder.transform(frame[self.__class__.cat_var]).toarray(),
            axis = 1
        )

    def scale(self, data):
        return self.standard_scaler.transform(data)

    def get_labels(self, frame):
        return frame[self.__class__.lab_var].to_numpy()

    def fit(self, frame):
        # Encoding
        self.one_hot_encoder.fit(frame[self.__class__.cat_var])
        data = self.encode(frame)
        # Scaling
        self.standard_scaler.fit(data)
        self.x = self.scale(data)
        # Training
        self.y = self.get_labels(frame)
        self.predictor.fit(self.x)

    def transform(self, frame):
        data = self.encode(frame)
        return self.scale(data)

    def predict(self, frame):
        if self.x is None or self.y is None:
            raise AssertionError('The model has not been trained yet.')
        data = self.transform(frame)
        regression = self.predictor.kneighbors(data, return_distance = True)
        num_entries = regression[0].shape[0]
        result = np.array([0.0 for _ in range(num_entries)])
        limit = (self.n_neighbors + 1) // 2
        for idx in range(num_entries):
            distances = regression[0][idx]
            labels = regression[1][idx]
            num_vals = 0
            sum_wgts = 0
            sum_vals = 0
            for k in range(self.n_neighbors):
                if self.y[labels[k]] != -1:
                    num_vals += 1
                    if distances[k] != 0:
                        sum_wgts += 1.0 / distances[k]
                        sum_vals += (1.0 / distances[k]) * self.y[labels[k]]
                    else:
                        sum_wgts += 10 ** 12
                        sum_vals += (10 ** 12) * self.y[labels[k]]
                num_infs = k + 1 - num_vals
                if num_infs >= limit:
                    result[idx] = -1
                    break
                if num_vals >= limit:
                    result[idx] = sum_vals / sum_wgts
                    break
        return result

#%%
frame = pd.read_csv('TrainingData.csv')
frame_train, frame_test = train_test_split(frame, random_state = 144, test_size = 0.2)

#%%
# Data exploration
x = frame[SuspendedUsersModel.con_var].to_numpy()
y = frame['TiempoTotalSuspension'].to_numpy()

f_scores = f_regression(x, y)
mut_info = mutual_info_regression(x, y)
for idx in range(len(SuspendedUsersModel.con_var)):
    print(SuspendedUsersModel.con_var[idx], f_scores[0][idx], mut_info[idx])

#%%
# Search for an appropiate Hyperparameter
for n_neighbors in range(1, 99 + 1, 2):
    model = SuspendedUsersModel(n_neighbors = n_neighbors)
    model.fit(frame_train)
    prediction_result = model.predict(frame_test)
    prediction_error = mean_squared_error(frame_test[model.lab_var], prediction_result)
    print((n_neighbors, prediction_error))

#%%
training_data = pd.read_csv('TrainingData.csv')
model = SuspendedUsersModel(n_neighbors = 41)
model.fit(training_data)

input_data = pd.read_csv('InputData.csv')
prediction_result = model.predict(input_data)

frame_out = pd.concat([
    input_data.reset_index(),
    pd.DataFrame({ 'TiempoTotalSuspensionEsperado': prediction_result })
], sort = False, axis = 1)

frame_out.to_csv('OutputData.csv')

# %%
