import numpy as np
from tensorflow.keras.datasets import mnist
from utilities import plot_metrics, train_model
import time
from keras.backend import clear_session
from MLP.models import model_sgd_l1, model_sgd_l2_case1, model_sgd_l2_case2, model_sgd_l2_case3, model_vanilla_case1, \
    model_vanilla_case2, compile_models

epochs = 10
num_features = 784  # data features (img shape: 28*28).
validation_split = 0.2

# Prepare MNIST data.
(x_train, y_train), (x_test, y_test) = mnist.load_data()
# Convert to float32.
x_train, x_test = np.array(x_train, np.float32), np.array(x_test, np.float32)
# Flatten images to 1-D vector of 784 features (28*28).
x_train, x_test = x_train.reshape([-1, num_features]), x_test.reshape([-1, num_features])
# Normalize images value from [0, 255] to [0, 1].
x_train, x_test = x_train / 255., x_test / 255.

batch_sizes = [1, 256, x_train.shape[0]]
total_training_times = []

if __name__ == "__main__":
    compile_models()

    for batch_size in [batch_sizes[1], batch_sizes[2]]:
        print("Training " + model_vanilla_case1.name + " for batch size = " + str(batch_size))
        train_model(model=model_vanilla_case1, x_train=x_train, y_train=y_train, validation_split=validation_split,
                    epochs=epochs, batch_size=batch_size, total_training_times=total_training_times)

        print("Training " + model_vanilla_case2.name + " for batch size = " + str(batch_size))
        train_model(model=model_vanilla_case2, x_train=x_train, y_train=y_train, validation_split=validation_split,
                    epochs=epochs, batch_size=batch_size, total_training_times=total_training_times)

        print("Training " + model_sgd_l1.name + " for batch size = " + str(batch_size))
        train_model(model=model_sgd_l1, x_train=x_train, y_train=y_train, validation_split=validation_split,
                    epochs=epochs, batch_size=batch_size, total_training_times=total_training_times)

        print("Training " + model_sgd_l2_case1.name + " for batch size = " + str(batch_size))
        train_model(model=model_sgd_l2_case1, x_train=x_train, y_train=y_train, validation_split=validation_split,
                    epochs=epochs, batch_size=batch_size, total_training_times=total_training_times)

        print("Training " + model_sgd_l2_case2.name + " for batch size = " + str(batch_size))
        train_model(model=model_sgd_l2_case2, x_train=x_train, y_train=y_train, validation_split=validation_split,
                    epochs=epochs, batch_size=batch_size, total_training_times=total_training_times)

        print("Training " + model_sgd_l2_case3.name + " for batch size = " + str(batch_size))
        train_model(model=model_sgd_l2_case3, x_train=x_train, y_train=y_train, validation_split=validation_split,
                    epochs=epochs, batch_size=batch_size, total_training_times=total_training_times)

        clear_session()

    print(total_training_times)
