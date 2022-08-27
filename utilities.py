import time
from matplotlib import pyplot as plt


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


def plot_metrics(model_history, model_name, batch_size):
    """"""
    plt.plot(model_history.history['accuracy'])
    plt.plot(model_history.history['val_accuracy'])
    plt.grid(b=True, color="#D3D3D3")
    plt.title('Model Accuracy for ' + model_name + "\n Batch Size = " + str(batch_size))
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Training', 'Evaluation'], loc='upper left')
    plt.savefig('MLP/plots/accuracy_' + model_name + ".png")
    plt.show()

    plt.plot(model_history.history['loss'])
    plt.plot(model_history.history['val_loss'])
    plt.grid(b=True, color='#D3D3D3')
    plt.title('Model Loss for ' + model_name + "\n Batch Size = " + str(batch_size))
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Training', 'Evaluation'], loc='upper left')
    plt.savefig('MLP/plots/loss_' + model_name + "_batch_size" + str(batch_size) + ".png")
    plt.show()
