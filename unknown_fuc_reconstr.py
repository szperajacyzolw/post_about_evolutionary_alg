'''Tekne Consulting blogpost --- teknecons.com'''


import plotly.graph_objects as go
import matplotlib.pyplot as plt
import sklearn.preprocessing as pre
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import train_test_split
from catboost import CatBoostRegressor
from tensorflow import keras
from tensorflow.keras import models, layers
import numpy as np
import pandas as pd
import itertools as it
import os
from hidden_function import hidden_fun


this_dir = os.path.dirname(os.path.abspath(__file__))
rg = np.random.default_rng(seed=1234)


'''random points generator'''


hidden_fvec = np.vectorize(hidden_fun)
randx = np.sort(rg.uniform(-10, 10, 25)).reshape(-1, 1)
randy = np.sort(rg.uniform(-10, 10, 25)).reshape(-1, 1)
rX, rY = np.meshgrid(randx, randy)
sampled_z = hidden_fvec(rX, rY).ravel().reshape(-1, 1)
randxy = np.concatenate((rX.ravel().reshape(-1, 1), rY.ravel().reshape(-1, 1)), axis=1)


'''scaling and splitting'''


xy_train, xy_test, z_train, z_test = train_test_split(
    randxy, sampled_z, test_size=0.33, random_state=123)

scaler_xy = pre.MinMaxScaler()
scaler_z = pre.MinMaxScaler()
scaled_xy_train = scaler_xy.fit_transform(xy_train)
scaled_z_train = scaler_z.fit_transform(z_train)
scaled_xy_test = scaler_xy.transform(xy_test)
scaled_z_test = scaler_z.transform(z_test)

'''splitting for tree validation'''


xy_val = xy_train[:100, :]
z_val = z_train[:100, :]
xy_train = xy_train[100:, :]
z_train = z_train[100:, :]


'''training of tree'''


t_model = CatBoostRegressor(num_trees=500, task_type='CPU', loss_function='RMSE',
                            thread_count=-1, verbose=100, random_seed=123)
t_model.fit(xy_train, z_train, eval_set=(xy_val, z_val))
t_model.save_model(os.path.join(this_dir, 'tree_approx_model.cbm'),
                   format="cbm",
                   export_parameters=None,
                   pool=None)


z_pred = t_model.predict(xy_test)
r2score = r2_score(z_test, z_pred)
mse = mean_squared_error(z_test, z_pred)
print(f'tree r2: {r2score}, mse: {mse}')


'''training neural network'''


initializer = keras.initializers.RandomUniform(minval=-1., maxval=1.)
n_model = models.Sequential()
n_model.add(keras.Input(shape=(2,)))
n_model.add(layers.Dense(40, activation='relu', kernel_initializer=initializer))
n_model.add(layers.Dense(30, activation='relu', kernel_initializer=initializer))
n_model.add(layers.Dense(1, activation='sigmoid'))
n_model.compile(loss='mse', optimizer='adam')
history = n_model.fit(xy_train, z_train, batch_size=2, epochs=350, validation_split=0.2)
model.save('neural_approx_model')

z_pred = model.predict(scaled_xy_test)
r2score = r2_score(scaled_z_test, z_pred)
mse = mean_squared_error(scaled_z_test, z_pred)
print(f'neural r2: {r2score}, mse: {mse}')

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.legend(['train', 'test'], loc='upper left')
plt.show()


'''re-creating hidden function shape'''


rg = np.random.default_rng(seed=12345)
randx = np.sort(rg.uniform(-10, 10, 100))
randy = np.sort(rg.uniform(-10, 10, 100))
rX, rY = np.meshgrid(randx, randy)
plot_data = np.array([rX.ravel(), rY.ravel()]).T
X = plot_data[:, 0].reshape(100, 100)
Y = plot_data[:, 1].reshape(100, 100)
predict_full_t = t_model.predict(plot_data).reshape(-1, 1)
Zt = predict_full_t.reshape([100, 100])

fig1 = go.Figure(
    data=[go.Surface(z=Zt, x=X, y=Y)])
fig1.show()
fig1.write_html(os.path.join(this_dir, 'tree_fun_reconst.html'))

predict_full_n = n_model.predict(plot_data).reshape(-1, 1)
Zn = predict_full_n.reshape([100, 100])

fig2 = go.Figure(
    data=[go.Surface(z=Zn, x=X, y=Y)])
fig2.show()
fig2.write_html(os.path.join(this_dir, 'neural_fun_reconst.html'))
