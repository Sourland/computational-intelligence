import os

from matplotlib import pyplot as plt

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
import keras_tuner as kt

from tensorflow import keras
from tensorflow.keras.datasets import mnist
from tensorflow.keras.callbacks import EarlyStopping

from keras import layers
from keras.utils import to_categorical

from metrics import precision, recall, f_measure
from utilities import plot_metrics, plot_confusion_matrix
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

import seaborn as sn

num_features = 784  # data features (img shape: 28*28).
validation_split = 0.2
num_classes = 10
RANDOM_VARIABLE = 42
# Prepare MNIST data.
(x_train, y_train), (x_test, y_test) = mnist.load_data()
# Convert to float32.
# Flatten images to 1-D vector of 784 features (28*28).
x_train, x_test = x_train.reshape([-1, num_features]), x_test.reshape([-1, num_features])
# Normalize images value from [0, 255] to [0, 1].
x_train.astype('float32')
y_test.astype('float32')

x_train, x_test = x_train / 255., x_test / 255.

y_train, y_test = to_categorical(y_train, num_classes), to_categorical(y_test, num_classes)

print(x_test.shape)
print(x_train.shape)
print(y_test.shape)
print(y_train.shape)


def build_model(hp):
    model = keras.Sequential()

    layer1_units = hp.Choice('layer1_units', values=[64, 128])
    layer2_units = hp.Choice('layer2_units', values=[128, 256])
    learning_rates = hp.Choice('learning_rate', values=[1e-1, 1e-2, 1e-3])
    regularizer = keras.regularizers.L2(l2=hp.Choice('l2_coeff', values=[1e-1, 1e-3, 1e-6]))

    initializer = keras.initializers.HeNormal()

    network_layers = [
        layers.Dense(units=layer1_units, activation=tf.nn.relu,
                     kernel_regularizer=regularizer,
                     kernel_initializer=initializer,
                     name="Layer1"),
        layers.Dense(units=layer2_units, activation=tf.nn.relu,
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
                  metrics=['accuracy', precision, recall, f_measure]
                  )

    return model


total_epochs = 5

tuner = kt.RandomSearch(
    build_model,
    objective=kt.Objective("f_measure", direction='max'),
    directory='keras_tuner_dir',
    project_name='mlp_tuning',
    overwrite=True
)

early_stopping = EarlyStopping(
    monitor="loss",
    patience=200,
    restore_best_weights=True)

tuner.search(x_train, y_train,
             epochs=total_epochs,
             validation_split=0.2,
             callbacks=[early_stopping],
             )

optimal_hyperparameters = tuner.get_best_hyperparameters(num_trials=1)[0]
layer1_units = optimal_hyperparameters.get('layer1_units')
layer2_units = optimal_hyperparameters.get('layer2_units')
learning_rate = optimal_hyperparameters.get('learning_rate')
l2_coeff = optimal_hyperparameters.get('l2_coeff')

tuned_model = keras.Sequential()

network_layers = [
    layers.Dense(units=layer1_units, activation=tf.nn.relu,
                 kernel_regularizer=keras.regularizers.L2(l2=l2_coeff),
                 kernel_initializer=keras.initializers.HeNormal(),
                 name="Layer1"),
    layers.Dense(units=layer2_units, activation=tf.nn.relu,
                 kernel_regularizer=keras.regularizers.L2(l2=l2_coeff),
                 kernel_initializer=keras.initializers.HeNormal(),
                 name="Layer2"),
    layers.Dense(units=10,
                 activation=tf.nn.softmax,
                 name="LayerOut")
]

for layer in network_layers:
    tuned_model.add(layer)

tuned_model.compile(optimizer=tf.optimizers.RMSprop(learning_rate=learning_rate),
                    loss=tf.keras.losses.CategoricalCrossentropy(),
                    metrics=['accuracy', precision, recall, f_measure]
                    )

history = tuned_model.fit(x_train, y_train, epochs=10, validation_split=0.2)
plot_metrics(history, "Tuned MLP network")

loss, accuracy, f1_score, model_precision, model_recall = tuned_model.evaluate(x_test, y_test, verbose=0)

print('Test loss:', loss)
print('Test accuracy:', accuracy)
print('Test f1_score:', f1_score)
print('Test model precision:', model_precision)
print('Test model recall:', model_recall)

y_pred = tuned_model.predict(y_test)
result = tf.math.confusion_matrix(y_test, y_pred)
total_classes = [i for i in range(0, 10)]
plot_confusion_matrix(result, classes=total_classes)
