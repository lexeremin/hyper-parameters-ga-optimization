from sklearn.metrics import confusion_matrix, multilabel_confusion_matrix, ConfusionMatrixDisplay
import numpy as np
import matplotlib.pyplot as plt


def make_confusion_matrix(y_true, y_pred, classes=None, figsize=(10, 10), text_size=15, fname=False):

    # Create the confustion matrix
    cm = confusion_matrix(y_true, y_pred)

    # Are there a list of classes?
    if classes:
        labels = classes
    else:
        labels = np.arange(cm.shape[0])
    # print(labels)

    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    disp.plot()
    plt.title('Confusion matrix')
    plt.xlabel('Predicted classes')
    plt.ylabel('True classes')
    if fname:
        plt.savefig('./imgs/'+fname+'cm_opt.png')
    plt.show()
