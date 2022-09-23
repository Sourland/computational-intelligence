from matplotlib import pyplot as plt
import numpy as np
from RBF.data_processing import k_means_cluster, calculate_center_distance


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
        plt.savefig('MLP/plots/accuracy_' + model_name + ".png")
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
        plt.savefig('MLP/plots/loss_' + model_name + "_batch_size" + str(batch_size) + ".png")
        plt.show()

    else:
        title_1 = r'$R^2$' + ' for ' + model_name

        plt.plot(model_history.history['rho_squared'])
        plt.plot(model_history.history['val_rho_squared'])
        plt.title(title_1)
        plt.ylabel('coefficient of determination')
        plt.xlabel('epoch')
        plt.legend(['train', 'validation'], loc='upper left')
        plt.savefig('plots/r2_' + model_name + ".png")
        plt.show()

        title_2 = 'Model Loss for ' + model_name
        plt.plot(model_history.history['loss'])
        plt.plot(model_history.history['val_loss'])
        plt.title(title_2)
        plt.ylabel('cost')
        plt.xlabel('epoch')
        plt.legend(['train', 'validation'], loc='upper left')
        plt.savefig('plots/loss_' + model_name + ".png")
        plt.show()


def calculate_rbf_elements(data, size):
    center, x_tran = k_means_cluster(data, size)
    cluster_center_max_dif = max(calculate_center_distance(center))

    sigma = cluster_center_max_dif / np.sqrt(2 * size)
    betas = 1 / (2 * sigma ** 2)
    return center, betas

# def plot_confusion_matrix(confusion_matrix, classes,
#                           normalize=False,
#                           title='Confusion matrix',
#                           color_map=plt.confusion_matrix.Blues):
#     plt.imshow(confusion_matrix, interpolation='nearest', confusion_matrixap=color_map)
#     plt.title(title)
#     plt.colorbar()
#     tick_marks = np.arange(len(classes))
#     plt.xticks(tick_marks, classes, rotation=45)
#     plt.yticks(tick_marks, classes)
#
#     if normalize:
#         confusion_matrix = confusion_matrix.astype('float') / confusion_matrix.sum(axis=1)[:, np.newaxis]
#         print("Normalized confusion matrix")
#     else:
#         print('Confusion matrix, without normalization')
#
#     thresh = confusion_matrix.max() / 2.
#     for i, j in itertools.product(range(confusion_matrix.shape[0]), range(confusion_matrix.shape[1])):
#         plt.text(j, i, confusion_matrix[i, j],
#                  horizontalalignment="center",
#                  color="white" if confusion_matrix[i, j] > thresh else "black")
#
#     plt.tight_layout()
#     plt.ylabel('True label')
#     plt.xlabel('Predicted label')
