from tensorflow.keras import Model, layers, regularizers
import tensorflow as tf


RANDOM_VARIABLE = 42
# Network parameters.
n_hidden_1 = 128  # 1st layer number of neurons.
n_hidden_2 = 256  # 2nd layer number of neurons.
num_classes = 10

layers_vanilla = [
    layers.Dense(n_hidden_1, activation=tf.nn.relu, name="Layer1"),
    layers.Dense(n_hidden_2, activation=tf.nn.relu, name="Layer2"),
    layers.Dense(num_classes, activation=tf.nn.softmax, name="LayerOut"),
]
alpha_l1 = 0.01

layers_l1 = [
    layers.Dense(n_hidden_1, activation=tf.nn.relu, kernel_regularizer=regularizers.L1(l1=alpha_l1), name="Layer1"),
    layers.Dense(n_hidden_2, activation=tf.nn.relu, kernel_regularizer=regularizers.L1(l1=alpha_l1), name="Layer2"),
    layers.Dense(num_classes, activation=tf.nn.softmax, name="LayerOut"),
]

layers_l2_case1 = [
    layers.Dense(units=n_hidden_1,
                 activation=tf.nn.relu,
                 kernel_regularizer=regularizers.L2(l2=1e-1),
                 kernel_initializer=tf.keras.initializers.RandomNormal(mean=10, stddev=1, seed=RANDOM_VARIABLE),
                 name="Layer1"),
    layers.Dense(units=n_hidden_2,
                 activation=tf.nn.relu,
                 kernel_regularizer=regularizers.L2(l2=1e-1),
                 kernel_initializer=tf.keras.initializers.RandomNormal(mean=10, stddev=1, seed=RANDOM_VARIABLE),
                 name="Layer2"),
    layers.Dense(units=num_classes,
                 activation=tf.nn.softmax,
                 name="LayerOut")
]
layers_l2_case2 = [
    layers.Dense(units=n_hidden_1,
                 activation=tf.nn.relu,
                 kernel_regularizer=regularizers.L2(l2=1e-2),
                 kernel_initializer=tf.keras.initializers.RandomNormal(mean=10, stddev=1, seed=RANDOM_VARIABLE),
                 name="Layer1"),
    layers.Dense(units=n_hidden_2,
                 activation=tf.nn.relu,
                 kernel_regularizer=regularizers.L2(l2=1e-2),
                 kernel_initializer=tf.keras.initializers.RandomNormal(mean=10, stddev=1, seed=RANDOM_VARIABLE),
                 name="Layer2"),
    layers.Dense(units=num_classes,
                 activation=tf.nn.softmax,
                 name="LayerOut")
]
layers_l2_case3 = [
    layers.Dense(units=n_hidden_1,
                 activation=tf.nn.relu,
                 kernel_regularizer=regularizers.L2(l2=1e-3),
                 kernel_initializer=tf.keras.initializers.RandomNormal(mean=10, stddev=1, seed=RANDOM_VARIABLE),
                 name="Layer1"),
    layers.Dense(units=n_hidden_2,
                 activation=tf.nn.relu,
                 kernel_regularizer=regularizers.L2(l2=1e-3),
                 kernel_initializer=tf.keras.initializers.RandomNormal(mean=10, stddev=1, seed=RANDOM_VARIABLE),
                 name="Layer2"),
    layers.Dense(units=num_classes,
                 activation=tf.nn.softmax,
                 name="LayerOut")
]