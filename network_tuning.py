from tensorflow import keras
from keras import Model, layers, regularizers
from keras.utils import to_categorical
from metrics import accuracy, precision, recall, f_measure
import tensorflow as tf
import keras_tuner as kt
from tensorflow.keras.datasets import mnist
from tensorflow.keras.callbacks import EarlyStopping
import numpy as np
from sklearn.model_selection import train_test_split

num_features = 784  # data features (img shape: 28*28).
validation_split = 0.2
num_classes = 10
RANDOM_VARIABLE = 42
# Prepare MNIST data.
(x_train, y_train), (x_test, y_test) = mnist.load_data()
# Convert to float32.
x_train, x_test = np.array(x_train, np.float32), np.array(x_test, np.float32)
# Flatten images to 1-D vector of 784 features (28*28).
x_train, x_test = x_train.reshape([-1, num_features]), x_test.reshape([-1, num_features])
# Normalize images value from [0, 255] to [0, 1].
x_train, x_test = x_train / 255., x_test / 255.

y_train, y_test = to_categorical(y_train, num_classes), to_categorical(y_test, num_classes)

x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=validation_split,
                                                  random_state=RANDOM_VARIABLE)
print(x_test.shape)
print(x_train.shape)
print(x_val.shape)
print(y_test.shape)
print(y_train.shape)
print(y_val.shape)


def build_model(hp):
    model = keras.Sequential()

    layer1_total_neurons = hp.Choice('layer1', values=[64, 128])
    layer2_total_neurons = hp.Choice('layer2', values=[128, 256])
    learning_rates = hp.Choice('learning_rate', values=[1e-1, 1e-2, 1e-3])
    regularizer = keras.regularizers.L2(l2=hp.Choice('l2', values=[1e-1, 1e-3, 1e-6]))

    initializer = keras.initializers.HeNormal()

    network_layers = [
        layers.Dense(units=layer1_total_neurons, activation=tf.nn.relu, input_shape=(784,),
                     kernel_regularizer=regularizer,
                     kernel_initializer=initializer,
                     name="Layer1"),
        layers.Dense(units=layer2_total_neurons, activation=tf.nn.relu,
                     kernel_regularizer=regularizer,
                     kernel_initializer=initializer,
                     name="Layer2"),
        layers.Dense(units=10,
                     activation=tf.nn.softmax,
                     name="LayerOut")
    ]

    for layer in network_layers:
        model.add(layer)

    model.compile(optimizer=tf.optimizers.RMSprop(learning_rate=learning_rates),
                  loss=tf.keras.losses.CategoricalCrossentropy(),
                  metrics=[accuracy, precision, recall, f_measure]
                  )

    return model


tuning_epochs = 1000

tuner = kt.RandomSearch(build_model,
                        objective=kt.Objective('f1_m', direction='max'),
                        directory='keras_tuner_dir',
                        project_name='mlp_tuning'
                        )

es = EarlyStopping(
    monitor="f1_m",
    patience=200,
    restore_best_weights=True)

tuner.search(x=x_train, y=y_train,
             validation_split=0.2,
             batch_size=256,
             callbacks=[es],
             epochs=tuning_epochs)
