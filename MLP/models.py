import tensorflow as tf
import tensorflow.keras as keras
from .layers import layers_vanilla, layers_l1, layers_l2_case1, layers_l2_case2, layers_l2_case3

model_vanilla = keras.Sequential(layers_vanilla)
model_sgd_l1 = keras.Sequential(layers_l1)
model_sgd_l2_case1 = keras.Sequential(layers_l2_case1)
model_sgd_l2_case2 = keras.Sequential(layers_l2_case2)
model_sgd_l2_case3 = keras.Sequential(layers_l2_case3)
learning_rate = 0.001


model_vanilla.compile(optimizer=tf.optimizers.RMSprop(learning_rate = learning_rate, rho = 0.9),
              loss=tf.keras.losses.SparseCategoricalCrossentropy(),
              metrics=['accuracy']
              )

model_sgd_l1.compile(optimizer=tf.optimizers.SGD(learning_rate = learning_rate),
                loss=tf.keras.losses.SparseCategoricalCrossentropy(),
                metrics=['accuracy'])

model_sgd_l2_case1.compile(optimizer=tf.optimizers.SGD(learning_rate = learning_rate),
                loss=tf.keras.losses.SparseCategoricalCrossentropy(),
                metrics=['accuracy'])

model_sgd_l2_case1.compile(optimizer=tf.optimizers.SGD(learning_rate = learning_rate),
                loss=tf.keras.losses.SparseCategoricalCrossentropy(),
                metrics=['accuracy'])

model_sgd_l2_case1.compile(optimizer=tf.optimizers.SGD(learning_rate = learning_rate),
                loss=tf.keras.losses.SparseCategoricalCrossentropy(),
                metrics=['accuracy'])