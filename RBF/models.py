import keras
from keras import datasets
import numpy as np
import tensorflow as tf
from keras import datasets, layers, optimizers, losses, metrics
from RBFLayer import RBFLayer
from metrics import rho_squared
from utilities import plot_metrics
from sklearn.preprocessing import normalize
from kmeans_initializer import InitCentersKMeans

# Importing Dataset
(x_train, y_train), (x_test, y_test) = datasets.boston_housing.load_data(path="boston_housing.npz", test_split=.25)
train_data_size = np.max(x_train.shape)
rbf_layer_sizes = np.round(np.array([0.1 * train_data_size, 0.5 * train_data_size, 0.9 * train_data_size])).astype(int)
dense_layer_size = 128
output_layer_size = 1
learning_rate = 1e-3

x_train = normalize(x_train)
x_test = normalize(x_test)

rbf_model_1 = keras.Sequential(name='RBF-' + str(rbf_layer_sizes[0]) + '-neurons-model')
rbf_model_1.add(RBFLayer(rbf_layer_sizes[0], initializer=InitCentersKMeans(x_train), input_shape=(13,)))
rbf_model_1.add(layers.Dense(dense_layer_size, ))
rbf_model_1.add(layers.Dense(output_layer_size, ))

rbf_model_1.compile(optimizer=optimizers.SGD(learning_rate=learning_rate), loss=losses.MeanSquaredError(),
                    metrics=[metrics.RootMeanSquaredError(), rho_squared])

rbf_model_2 = keras.Sequential(name='RBF-' + str(rbf_layer_sizes[1]) + '-neurons-model')
rbf_model_2.add(RBFLayer(rbf_layer_sizes[1], initializer=InitCentersKMeans(x_train), input_shape=(13,)))
rbf_model_2.add(layers.Dense(dense_layer_size, ))
rbf_model_2.add(layers.Dense(output_layer_size, ))

rbf_model_2.compile(optimizer=optimizers.SGD(learning_rate=learning_rate), loss=losses.MeanSquaredError(),
                    metrics=[metrics.RootMeanSquaredError(), rho_squared])

rbf_model_3 = keras.Sequential(name='RBF-' + str(rbf_layer_sizes[2]) + '-neurons-model')
rbf_model_3.add(RBFLayer(rbf_layer_sizes[2], initializer=InitCentersKMeans(x_train), input_shape=(13,)))
rbf_model_3.add(layers.Dense(dense_layer_size, ))
rbf_model_3.add(layers.Dense(output_layer_size, ))

rbf_model_3.compile(optimizer=optimizers.SGD(learning_rate=learning_rate), loss=losses.MeanSquaredError(),
                    metrics=[metrics.RootMeanSquaredError(), rho_squared])

models_r2 = []
models_r2_val = []
models_rmse = []
models_rmse_val = []

history_1 = rbf_model_1.fit(x_train, y_train, epochs=100, validation_split=0.2)
plot_metrics(model_history=history_1, model_name=rbf_model_1.name, model_class='rbf')

models_r2.append(history_1.history['rho_squared'][-1])
models_r2_val.append(history_1.history['val_rho_squared'][-1])
models_rmse.append(history_1.history['root_mean_squared_error'][-1])
models_rmse_val.append(history_1.history['val_root_mean_squared_error'][-1])

history_2 = rbf_model_2.fit(x_train, y_train, epochs=100, validation_split=0.2)
plot_metrics(model_history=history_2, model_name=rbf_model_2.name, model_class='rbf')

models_r2.append(history_2.history['rho_squared'][-1])
models_r2_val.append(history_2.history['val_rho_squared'][-1])
models_rmse.append(history_2.history['root_mean_squared_error'][-1])
models_rmse_val.append(history_2.history['val_root_mean_squared_error'][-1])

history_3 = rbf_model_3.fit(x_train, y_train, epochs=100, validation_split=0.2)
plot_metrics(model_history=history_3, model_name=rbf_model_3.name, model_class='rbf')

models_r2.append(history_3.history['rho_squared'][-1])
models_r2_val.append(history_3.history['val_rho_squared'][-1])
models_rmse.append(history_3.history['root_mean_squared_error'][-1])
models_rmse_val.append(history_3.history['val_root_mean_squared_error'][-1])

i = 1
for r2, val_r2, rmse, val_rmse in zip(models_r2, models_r2_val, models_rmse, models_rmse_val):
    print(
        f"Model {i} r2 = {r2}, val_r2 = {val_r2}, RMSE = {rmse} and val_RMSE = {val_rmse}".format(i=i, r2=r2,
                                                                                                  val_r2=val_r2,
                                                                                                  rmse=rmse,
                                                                                                  val_rmse=val_rmse))
    i += 1
