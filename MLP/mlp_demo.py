import numpy as np
from tensorflow.keras.datasets import mnist
from utilities import plot_metrics
from time import time
from keras.backend import clear_session
from MLP.models import model_default, model_sgd, model_sgd_l1, model_sgd_l2_case1, model_sgd_l2_case2, model_sgd_l2_case3, \
    model_rmsprop_case1, model_rmsprop_case2, compile_models

epochs = 100
num_features = 784  # data features (img shape: 28*28).
validation_split = 0.2

# Prepare MNIST data.
(x_train, y_train), (x_test, y_test) = mnist.load_data()
# Convert to float32.
x_train, x_test = np.array(x_train, np.float32), np.array(x_test, np.float32)
print(x_train.shape)
# Flatten images to 1-D vector of 784 features (28*28).
x_train, x_test = x_train.reshape([-1, num_features]), x_test.reshape([-1, num_features])
# Normalize images value from [0, 255] to [0, 1].
x_train, x_test = x_train / 255., x_test / 255.
total_training_times = []
if __name__ == "__main__":
    compile_models()
    for batch_size in [1, 256, np.floor(0.8 * x_train.shape[0]).astype(int)]:
        print("Training " + model_default.name + " for batch size = " + str(batch_size))
        start = time()
        history = model_default.fit(x_train, y_train, validation_split=validation_split, batch_size=batch_size,
                                          epochs=epochs)
        end = time()
        total_training_times.append(end - start)
        plot_metrics(history, model_default.name, batch_size=batch_size)
        clear_session()

    batch_size = 256
    print("Training " + model_rmsprop_case1.name + " for batch size = " + str(batch_size))
    start = time()
    history = model_rmsprop_case1.fit(x_train, y_train, validation_split=validation_split, batch_size=batch_size,
                                      epochs=epochs)
    end = time()
    total_training_times.append(end - start)
    plot_metrics(history, model_rmsprop_case1.name)

    print("Training " + model_rmsprop_case2.name + " for batch size = " + str(batch_size))
    start = time()
    history = model_rmsprop_case2.fit(x_train, y_train, validation_split=validation_split, batch_size=batch_size,
                                      epochs=epochs)
    end = time()
    total_training_times.append(end - start)
    plot_metrics(history, model_rmsprop_case2.name)

    print("Training " + model_sgd.name + " for batch size = " + str(batch_size))
    start = time()
    history = model_sgd.fit(x_train, y_train, validation_split=validation_split, batch_size=batch_size,
                                      epochs=epochs)
    end = time()
    total_training_times.append(end - start)
    plot_metrics(history, model_sgd.name)

    print("Training " + model_sgd_l1.name + " for batch size = " + str(batch_size))
    start = time()
    history = model_sgd_l1.fit(x_train, y_train, validation_split=validation_split, batch_size=batch_size,
                               epochs=epochs)
    end = time()
    total_training_times.append(end - start)
    plot_metrics(history, model_sgd_l1.name)

    print("Training " + model_sgd_l2_case1.name + " for batch size = " + str(batch_size))
    start = time()
    history = model_sgd_l2_case1.fit(x_train, y_train, validation_split=validation_split, batch_size=batch_size,
                                     epochs=epochs)
    end = time()
    total_training_times.append(end - start)
    plot_metrics(history, model_sgd_l2_case1.name)

    print("Training " + model_sgd_l2_case2.name + " for batch size = " + str(batch_size))
    start = time()
    history = model_sgd_l2_case2.fit(x_train, y_train, validation_split=validation_split, batch_size=batch_size,
                                     epochs=epochs)
    end = time()
    total_training_times.append(end - start)
    plot_metrics(history, model_sgd_l2_case2.name)

    print("Training " + model_sgd_l2_case3.name + " for batch size = " + str(batch_size))
    start = time()
    history = model_sgd_l2_case3.fit(x_train, y_train, validation_split=validation_split, batch_size=batch_size,
                                     epochs=epochs)
    end = time()
    total_training_times.append(end - start)
    plot_metrics(history, model_sgd_l2_case3.name)

    print(total_training_times)
