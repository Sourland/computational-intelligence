import tensorflow as tf
from tensorflow import keras
from .layers import layers_vanilla, layers_l1, layers_l2_case1, layers_l2_case2, layers_l2_case3

model_default = keras.Sequential(layers=layers_vanilla, name="default_network")
model_rmsprop_case1 = keras.Sequential(layers=layers_vanilla, name="RMSProp_case1")
model_rmsprop_case2 = keras.Sequential(layers=layers_vanilla, name="RMSProp_case2")
model_sgd = keras.Sequential(layers=layers_vanilla, name="SGD")
model_sgd_l1 = keras.Sequential(layers=layers_l1, name="L1_reg")
model_sgd_l2_case1 = keras.Sequential(layers=layers_l2_case1, name="L2_reg_1e-1")
model_sgd_l2_case2 = keras.Sequential(layers=layers_l2_case2, name="L2_reg_1e-3")
model_sgd_l2_case3 = keras.Sequential(layers=layers_l2_case3, name="L2_reg_1e-6")
learning_rate_sgd = 0.001
learning_rate_rmsprop = 0.01


def compile_models():
    model_default.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(),
                          metrics=['accuracy'])

    model_rmsprop_case1.compile(optimizer=tf.optimizers.RMSprop(learning_rate=learning_rate_rmsprop, rho=0.9),
                                loss=tf.keras.losses.SparseCategoricalCrossentropy(),
                                metrics=['accuracy']
                                )

    model_rmsprop_case2.compile(optimizer=tf.optimizers.RMSprop(learning_rate=learning_rate_rmsprop, rho=0.1),
                                loss=tf.keras.losses.SparseCategoricalCrossentropy(),
                                metrics=['accuracy']
                                )

    model_sgd.compile(optimizer=tf.optimizers.SGD(learning_rate=learning_rate_sgd),
                                loss=tf.keras.losses.SparseCategoricalCrossentropy(),
                                metrics=['accuracy'])

    model_sgd_l1.compile(optimizer=tf.optimizers.SGD(learning_rate=learning_rate_sgd),
                         loss=tf.keras.losses.SparseCategoricalCrossentropy(),
                         metrics=['accuracy']
                         )

    model_sgd_l2_case1.compile(optimizer=tf.optimizers.SGD(learning_rate=learning_rate_sgd),
                               loss=tf.keras.losses.SparseCategoricalCrossentropy(),
                               metrics=['accuracy']
                               )

    model_sgd_l2_case2.compile(optimizer=tf.optimizers.SGD(learning_rate=learning_rate_sgd),
                               loss=tf.keras.losses.SparseCategoricalCrossentropy(),
                               metrics=['accuracy']
                               )

    model_sgd_l2_case3.compile(optimizer=tf.optimizers.SGD(learning_rate=learning_rate_sgd),
                               loss=tf.keras.losses.SparseCategoricalCrossentropy(),
                               metrics=['accuracy']
                               )
