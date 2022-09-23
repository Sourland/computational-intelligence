import keras
from keras import datasets
import numpy as np
import tensorflow as tf
from keras import datasets, layers, optimizers, losses, metrics
from data_processing import k_means_cluster, calculate_center_distance
from RBFLayer import RBFLayer
from metrics import rho_squared
from utilities import plot_metrics
from sklearn.preprocessing import normalize

# Importing Dataset
(x_train, y_train), (x_test, y_test) = datasets.boston_housing.load_data(path="boston_housing.npz", test_split=.25)
train_data_size = np.max(x_train.shape)
rbf_layer_sizes = np.round(np.array([0.1 * train_data_size, 0.5 * train_data_size, 0.9 * train_data_size])).astype(int)
dense_layer_size = 128
output_layer_size = 1
learning_rate = 1e-3

x_train = normalize(x_train)
x_test = normalize(x_test)
rbf_layers = []

for size in rbf_layer_sizes:
    centers, x_tran = k_means_cluster(x_train, size)
    cluster_center_max_dif = max(calculate_center_distance(centers))

    sigma = cluster_center_max_dif / np.sqrt(2 * size)
    betas = 1 / (2 * sigma ** 2)
    rbf_layers.append(RBFLayer(size, initializer=tf.constant_initializer(centers), betas=betas, input_shape=(13,)))

rbf_model_1 = keras.Sequential(name='RBF-' + str(rbf_layer_sizes[0]) + '-neurons-model')
rbf_model_1.add(rbf_layers[0])
rbf_model_1.add(layers.Dense(dense_layer_size, ))
rbf_model_1.add(layers.Dense(output_layer_size, ))

rbf_model_1.compile(optimizer=optimizers.SGD(learning_rate=learning_rate), loss=losses.MeanSquaredError(),
                    metrics=[metrics.RootMeanSquaredError(), rho_squared])

rbf_model_2 = keras.Sequential(name='RBF-' + str(rbf_layer_sizes[1]) + '-neurons-model')
rbf_model_2.add(rbf_layers[1])
rbf_model_2.add(layers.Dense(dense_layer_size, ))
rbf_model_2.add(layers.Dense(output_layer_size, ))

rbf_model_2.compile(optimizer=optimizers.SGD(learning_rate=learning_rate), loss=losses.MeanSquaredError(),
                    metrics=[metrics.RootMeanSquaredError(), rho_squared])

rbf_model_3 = keras.Sequential(name='RBF-' + str(rbf_layer_sizes[2]) + '-neurons-model')
rbf_model_3.add(rbf_layers[2])
rbf_model_3.add(layers.Dense(dense_layer_size, ))
rbf_model_3.add(layers.Dense(output_layer_size, ))

rbf_model_3.compile(optimizer=optimizers.SGD(learning_rate=learning_rate), loss=losses.MeanSquaredError(),
                    metrics=[metrics.RootMeanSquaredError(), rho_squared])

history_1 = rbf_model_1.fit(x_train, y_train, epochs=100, batch_size=128, validation_split=0.2)
plot_metrics(model_history=history_1, model_name=rbf_model_1.name, model_class='rbf')
history_2 = rbf_model_2.fit(x_train, y_train, batch_size=128, epochs=100, validation_split=0.2)
plot_metrics(model_history=history_2, model_name=rbf_model_2.name, model_class='rbf')
history_3 = rbf_model_3.fit(x_train, y_train, batch_size=128, epochs=100, validation_split=0.2)
plot_metrics(model_history=history_3, model_name=rbf_model_3.name, model_class='rbf')
