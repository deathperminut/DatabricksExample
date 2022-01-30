#%%
import numpy as np
import pandas as pd
from scipy import stats as st

#%%
initial_state = pd.read_csv('InitialState.csv')
transitions = pd.read_csv('Transitions.csv')
average_debt = pd.read_csv('AverageDebt.csv')['DeudaPromedio'][0]
transitions['Probabilidad'] = transitions['Cantidad'] / transitions['Total']

#%%
vector = initial_state['Cantidad'].to_numpy()
matrix = np.zeros((5, 5))
for i in range(5):
    for j in range(5):
        si = i * 30
        sj = j * 30
        data = transitions[(transitions['RangoEdadMoraInicial'] == si) & (transitions['RangoEdadMoraFinal'] == sj)]
        if len(data) != 0:
            matrix[j, i] = st.gmean(data.loc[:, 'Probabilidad'])

for i in range(5):
    factor = sum(matrix[:, i])
    for j in range(5):
        matrix[j, i] = matrix[j, i] / factor

#%%
print(matrix)
print(vector)
print(vector * average_debt)
vector1 = matrix.dot(vector)
print(vector1)
print(vector1 * average_debt)
vector2 = matrix.dot(vector1)
print(vector2)
print(vector2 * average_debt)
vector3 = matrix.dot(vector2)
print(vector3)
print(vector3 * average_debt)

# %%
