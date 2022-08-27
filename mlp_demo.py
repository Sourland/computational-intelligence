import numpy as np
from tensorflow.keras.datasets import mnist
from utilities import plot_metrics
import time
from keras.backend import clear_session
from MLP.models import model_sgd_l1, model_sgd_l2_case1, model_sgd_l2_case2, model_sgd_l2_case3, model_vanilla_case1, \
    model_vanilla_case2, compile_models


epochs = 100
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




batch_size = [1, 256, x_train.shape[0]]
print(x_train)
total_training_times = []
if __name__ == "__main__":
    # start = time.time()
    # history = model_vanilla_case1.fit(x_train, y_train, validation_split=validation_split, batch_size=batch_size_1,
    #                                   epochs=epochs)
    # end = time.time()
    # total_training_times.append(end - start)
    # plot_metrics(history, model_vanilla_case1.name, batch_size_1)

    compile_models()

    start = time.time()
    history = model_vanilla_case1.fit(x_train, y_train, validation_split=validation_split, batch_size=batch_size[1],
                                      epochs=epochs)
    end = time.time()
    total_training_times.append(end - start)
    plot_metrics(history, model_vanilla_case1.name, batch_size[1])

    print("\n\n")

    clear_session()

    start = time.time()
    history = model_vanilla_case1.fit(x_train, y_train, validation_split=validation_split, batch_size=batch_size[2],
                                      epochs=epochs)
    end = time.time()
    total_training_times.append(end - start)
    plot_metrics(history, model_vanilla_case1.name, batch_size[2])
