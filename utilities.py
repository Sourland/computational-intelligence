from matplotlib import pyplot as plt
import numpy as np
from time import time


def convert_to_preferred_format(sec):
    """Converts seconds to Hours-Minutes-Seconds (HMS) format

    Args:
        sec (int): The time in seconds

    Returns:
        String: The time in HMS format
    """
    sec = sec % (24 * 3600)
    hour = sec // 3600
    sec %= 3600
    minutes = sec // 60
    sec %= 60
    return "%02d:%02d:%02d" % (hour, minutes, sec)


def plot_metrics(model_history, model_name, batch_size=None, model_class='mlp'):
    """"""

    assert model_class in ['mlp', 'rbf']
    if model_class == 'mlp':
        plt.plot(model_history.history['accuracy'])
        plt.plot(model_history.history['val_accuracy'])
        plt.grid(b=True, color="#D3D3D3")

        if batch_size is None:
            title = 'Model Accuracy for ' + model_name
        else:
            title = 'Model Accuracy for ' + model_name + "\n Batch Size = " + str(batch_size)

        plt.title(title)
        plt.ylabel('Accuracy')
        plt.xlabel('Epoch')
        plt.legend(['Training', 'Evaluation'], loc='upper left')
        plt.savefig('MLP/plots/accuracy_' + model_name + "_batchsize=" + str(batch_size) + ".png")
        plt.show()

        plt.plot(model_history.history['loss'])
        plt.plot(model_history.history['val_loss'])
        plt.grid(b=True, color='#D3D3D3')

        if batch_size is None:
            title = 'Model Loss for ' + model_name
        else:
            title = 'Model Loss for ' + model_name + "\n Batch Size = " + str(batch_size)

        plt.title(title)
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.legend(['Training', 'Evaluation'], loc='upper left')
        plt.savefig('MLP/plots/loss_' + model_name + "_batchsize=" + str(batch_size) + ".png")
        plt.show()

    else:
        title_1 = r'$R^2$' + ' for ' + model_name

        plt.plot(model_history.history['rho_squared'])
        plt.plot(model_history.history['val_rho_squared'])
        plt.title(title_1)
        plt.ylabel('coefficient of determination')
        plt.xlabel('epoch')
        plt.legend(['train', 'validation'], loc='upper left')
        plt.savefig('r2_' + model_name + ".png")
        plt.show()

        title_2 = 'Model Loss for ' + model_name
        plt.plot(model_history.history['loss'])
        plt.plot(model_history.history['val_loss'])
        plt.title(title_2)
        plt.ylabel('cost')
        plt.xlabel('epoch')
        plt.legend(['train', 'validation'], loc='upper left')
        plt.savefig('loss_' + model_name + ".png")
        plt.show()
