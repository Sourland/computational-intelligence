import tensorflow as tf
from tensorflow import keras
from .layers import layers_vanilla, layers_l1, layers_l2_case1, layers_l2_case2, layers_l2_case3

model_vanilla_case1 = keras.Sequential(layers=layers_vanilla, name="Vanilla_case1")
model_vanilla_case2 = keras.Sequential(layers=layers_vanilla, name="Vanilla_case2")
model_sgd_l1 = keras.Sequential(layers=layers_l1, name="L1_reg")
model_sgd_l2_case1 = keras.Sequential(layers=layers_l2_case1, name="L2_reg_1e-1")
model_sgd_l2_case2 = keras.Sequential(layers=layers_l2_case2, name="L2_reg_1e-2")
model_sgd_l2_case3 = keras.Sequential(layers=layers_l2_case3, name="L2_reg_1e-3")
learning_rate_sgd = 0.001
learning_rate_rmsprop = 0.01


def compile_models():
    model_vanilla_case1.compile(optimizer=tf.optimizers.RMSprop(learning_rate=learning_rate_rmsprop, rho=0.9),
                                loss=tf.keras.losses.SparseCategoricalCrossentropy(),
                                metrics=['accuracy']
                                )

    model_vanilla_case2.compile(optimizer=tf.optimizers.RMSprop(learning_rate=learning_rate_rmsprop, rho=0.1),
                                loss=tf.keras.losses.SparseCategoricalCrossentropy(),
                                metrics=['accuracy']
                                )

    model_sgd_l1.compile(optimizer=tf.optimizers.SGD(learning_rate=learning_rate_sgd),
                         loss=tf.keras.losses.SparseCategoricalCrossentropy(),
                         metrics=['accuracy']
                         )

    model_sgd_l2_case1.compile(optimizer=tf.optimizers.SGD(learning_rate=learning_rate_sgd),
                               loss=tf.keras.losses.SparseCategoricalCrossentropy(),
                               metrics=['accuracy']
                               )

    model_sgd_l2_case1.compile(optimizer=tf.optimizers.SGD(learning_rate=learning_rate_sgd),
                               loss=tf.keras.losses.SparseCategoricalCrossentropy(),
                               metrics=['accuracy']
                               )

    model_sgd_l2_case1.compile(optimizer=tf.optimizers.SGD(learning_rate=learning_rate_sgd),
                               loss=tf.keras.losses.SparseCategoricalCrossentropy(),
                               metrics=['accuracy']
                               )

