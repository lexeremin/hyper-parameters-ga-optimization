import matplotlib.pyplot as plt

# Plot the validation and training data separately
def plot_loss_curves(history, val=True):
  """
  Returns separate loss curves for training and validation metrics.
  """ 
  loss = history.history['loss']
  accuracy = history.history['accuracy']

  if val:
    val_accuracy = history.history['val_accuracy']
    val_loss = history.history['val_loss']


  epochs = range(len(history.history['loss']))

  # Plot loss
  plt.plot(epochs, loss, label='training_loss')
  if val:
    plt.plot(epochs, val_loss, label='val_loss')
  plt.title('Loss')
  plt.xlabel('Epochs')
  plt.legend()

  # Plot accuracy
  plt.figure()
  plt.plot(epochs, accuracy, label='training_accuracy')
  if val:
    plt.plot(epochs, val_accuracy, label='val_accuracy')
  plt.title('Accuracy')
  plt.xlabel('Epochs')
  plt.legend()
  plt.show()