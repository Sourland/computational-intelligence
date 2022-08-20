from __future__ import absolute_import, division, print_function
import tensorflow as tf
from tensorflow.keras import Model, layers, regularizers
from tensorflow.keras.datasets import mnist
from tensorflow.keras.callbacks import EarlyStopping
import numpy as np3
import matplotlib.pyplot as plt
import pandas as pd

RANDOM_VARIABLE = 42

# MNIST dataset parameters.
num_classes = 10  # total classes (0-9 digits).
num_features = 784  # data features (img shape: 28*28).
# Training parameters.
learning_rate = 0.1
training_steps = 2000
batch_size = 256
display_step = 100
# Network parameters.
n_hidden_1 = 128  # 1st layer number of neurons.
n_hidden_2 = 256  # 2nd layer number of neurons.

# Prepare MNIST data.
(x_train, y_train), (x_test, y_test) = mnist.load_data()
# Convert to float32.
x_train, x_test = np.array(x_train, np.float32), np.array(x_test, np.float32)
# Flatten images to 1-D vector of 784 features (28*28).
x_train, x_test = x_train.reshape([-1, num_features]), x_test.reshape([-1, num_features])
# Normalize images value from [0, 255] to [0, 1].
x_train, x_test = x_train / 255., x_test / 255.

train_data = tf.data.Dataset.from_tensor_slices((x_train, y_train))
train_data = train_data.repeat().shuffle(5000).batch(batch_size).prefetch(1)


class NeuralNetwork(Model):
    def __init__(self, custom_weights=False, regularization=(None, None), dropout=(False, 0)):
        """

        Args:
            custom_weights:
            regularization:
            dropout:
        """
        super(NeuralNetwork, self).__init__()
        if regularization[0] == 'L1':
            alpha = regularization[1]
            self.fc1 = layers.Dense(n_hidden_1, input_shape=(num_features,), activation=tf.nn.relu,
                                    kernel_regularizer=regularizers.l1(alpha))
            if dropout[0]:
                self.drop_layer = layers.Dropout(dropout[1])

            self.fc2 = layers.Dense(n_hidden_2, activation=tf.nn.relu, kernel_regularizer=regularizers.l1(alpha))

        elif regularization[0] == 'L2':
            alpha = regularization[1]
            self.fc1 = layers.Dense(n_hidden_1, input_shape=(num_features,), activation=tf.nn.relu,
                                    kernel_regularizer=regularizers.l2(alpha))
            if dropout[0]:
                self.drop_layer = layers.Dropout(dropout[1])

            self.fc2 = layers.Dense(n_hidden_2, activation=tf.nn.relu, kernel_regularizer=regularizers.l2(alpha))
        else:
            self.fc1 = layers.Dense(n_hidden_1, input_shape=(num_features,), activation=tf.nn.relu)
            self.fc2 = layers.Dense(n_hidden_2, activation=tf.nn.relu)
            self.out = layers.Dense(num_classes, activation=tf.nn.softmax)

    def call(self, x, is_training=False):
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.out(x)
        if not is_training:
            # tf cross entropy expect logits without softmax, so only
            # apply softmax when not training.
            x = tf.nn.softmax(x)
        return x


neural_net = NeuralNetwork(regularization='L1')


def cross_entropy_loss(x, y):
    """
    Calculates the cross entropy loss between x and y
    Args:
        x:
        y:

    Returns:

    """
    # Convert labels to int 64 for tf cross-entropy function.
    y = tf.cast(y, tf.int64)
    # Apply softmax to logits and compute cross-entropy.
    loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=x)
    # Average loss across the batch.
    return tf.reduce_mean(loss)
    # Accuracy metric.


def accuracy(y_pred, y_true):
    """

    Args:
        y_pred:
        y_true:

    Returns:

    """
    # Predicted class is the index of the highest score in prediction vector (i.e. argmax).
    correct_prediction = tf.equal(tf.argmax(y_pred, 1), tf.cast(y_true, tf.int64))
    return tf.reduce_mean(tf.cast(correct_prediction, tf.float32), axis=-1)
    # Stochastic gradient descent optimizer.


SGD_optimizer = tf.optimizers.SGD(learning_rate)
RMSProp_optimizer1 = tf.optimizers.RMSprop(learning_rate=0.001, rho=0.1)
RMSProp_optimizer2 = tf.optimizers.RMSprop(learning_rate=0.001, rho=0.9)


# Optimization process.
def run_optimization(x, y, model, optimizer):
    """

    Args:
        x:
        y:
        model:
        optimizer:

    Returns:

    """
    # Wrap computation inside a GradientTape for automatic differentiation.
    with tf.GradientTape() as g:
        # Forward pass.
        prediction = model(x, is_training=True)
        # Compute loss.
        loss = cross_entropy_loss(prediction, y)
        # Variables to update, i.e. trainable variables.
        trainable_variables = model.trainable_variables
        # Compute gradients.
        gradients = g.gradient(loss, trainable_variables)
        # Update W and b following gradients.
        optimizer.apply_gradients(zip(gradients, trainable_variables))


for step, (batch_x, batch_y) in enumerate(train_data.take(training_steps), 1):
    # Run the optimization to update W and b values.
    run_optimization(batch_x, batch_y, neural_net, RMSProp_optimizer1)

    if step % display_step == 0:
        pred = neural_net(batch_x, is_training=True)
        loss = cross_entropy_loss(pred, batch_y)
        acc = accuracy(pred, batch_y)
        print("step: %i, loss: %f, accuracy: %f" % (step, loss, acc))
