import numpy as np
from tensorflow.keras.datasets import mnist
from MLP.models import model_sgd_l1, model_sgd_l2_case1, model_sgd_l2_case2, model_sgd_l2_case3, model_vanilla
from count_time import convert_to_preferred_format
import time
batch_size_1 = 1
batch_size_2 = 256
batch_size_3 = 60000

num_features = 784  # data features (img shape: 28*28).

# Prepare MNIST data.
(x_train, y_train), (x_test, y_test) = mnist.load_data()
# Convert to float32.
x_train, x_test = np.array(x_train, np.float32), np.array(x_test, np.float32)
# Flatten images to 1-D vector of 784 features (28*28).
x_train, x_test = x_train.reshape([-1, num_features]), x_test.reshape([-1, num_features])
# Normalize images value from [0, 255] to [0, 1].
x_train, x_test = x_train / 255., x_test / 255.

if __name__ == "__main__":

    start = time.time()
    model_vanilla.fit(x_train, y_train, batch_size = batch_size_1, epochs = 100)
    end = time.time()
    print("Total training time: " + convert_to_preferred_format(end-start))

    start = time.time()
    model_vanilla.fit(x_train, y_train, batch_size = batch_size_2, epochs = 100)
    end = time.time()
    print("Total training time: " + convert_to_preferred_format(end-start))

    start = time.time()
    model_vanilla.fit(x_train, y_train, batch_size = batch_size_3, epochs = 100)
    end = time.time()
    print("Total training time: " + convert_to_preferred_format(end-start))



