from sklearn.metrics import confusion_matrix, multilabel_confusion_matrix, ConfusionMatrixDisplay
import numpy as np
import matplotlib.pyplot as plt
import itertools

def make_confusion_matrix(y_true, y_pred, classes=None, figsize=(10, 10), text_size=15): 
  """Makes a labelled confusion matrix comparing predictions and ground truth labels.

  If classes is passed, confusion matrix will be labelled, if not, integer class values
  will be used.

  Args:
    y_true: Array of truth labels (must be same shape as y_pred).
    y_pred: Array of predicted labels (must be same shape as y_true).
    classes: Array of class labels (e.g. string form). If `None`, integer labels are used.
    figsize: Size of output figure (default=(10, 10)).
    text_size: Size of output figure text (default=15).
  
  Returns:
    A labelled confusion matrix plot comparing y_true and y_pred.

  Example usage:
    make_confusion_matrix(y_true=test_labels, # ground truth test labels
                          y_pred=y_preds, # predicted labels
                          classes=class_names, # array of class label names
                          figsize=(15, 15),
                          text_size=10)
  """  
  # Create the confustion matrix
  cm = confusion_matrix(y_true, y_pred)
  # cm_norm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis] # normalize it
  # n_classes = cm.shape[0] # find the number of classes we're dealing with

  # # Plot the figure and make it pretty
  # fig, ax = plt.subplots(figsize=figsize)
  # cax = ax.matshow(cm, cmap=plt.cm.Blues) # colors will represent how 'correct' a class is, darker == better
  # fig.colorbar(cax)

  # Are there a list of classes?
  if classes:
    labels = classes
  else:
    labels = np.arange(cm.shape[0])
  print(labels)
  # # Label the axes
  # ax.set(title="Confusion Matrix",
  #        xlabel="Predicted label",
  #        ylabel="True label",
  #        xticks=np.arange(n_classes), # create enough axis slots for each class
  #        yticks=np.arange(n_classes), 
  #        xticklabels=labels, # axes will labeled with class names (if they exist) or ints
  #        yticklabels=labels)
  
  # # Make x-axis labels appear on bottom
  # ax.xaxis.set_label_position("bottom")
  # ax.xaxis.tick_bottom()

  # # Set the threshold for different colors
  # threshold = (cm.max() + cm.min()) / 2.

  # # Plot the text on each cell
  # for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
  #   plt.text(j, i, f"{cm[i, j]} ({cm_norm[i, j]*100:.1f}%)",
  #            horizontalalignment="center",
  #            color="white" if cm[i, j] > threshold else "black",
  #            size=text_size)
  
  disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
  disp.plot()
  plt.show()