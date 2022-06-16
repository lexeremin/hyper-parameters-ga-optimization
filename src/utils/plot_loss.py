import matplotlib.pyplot as plt

# Plot the validation and training data separately
def plot_loss_curves(history, val=True, binary=False, fname=False):
  """
  Returns separate loss curves for training and validation metrics.
  """ 
  loss = history.history['loss']
  if binary:
    accuracy = history.history['binary_accuracy']
  else:
    accuracy = history.history['sparse_categorical_accuracy']

  if val:
    if binary:
      val_accuracy = history.history['val_binary_accuracy']
    else:  
      val_accuracy = history.history['val_sparse_categorical_accuracy']
    val_loss = history.history['val_loss']


  epochs = range(len(history.history['loss']))

  # Plot loss
  plt.plot(epochs, loss, label='Обучающая выборка')
  if val:
    plt.plot(epochs, val_loss, label='Тестовая выборка')
  plt.title('Функция потерь')
  plt.xlabel('число эпох')
  plt.legend()
  # Plot accuracy
  if fname:
    plt.savefig('./imgs/'+fname+'loss_opt.png')
  plt.figure()
  plt.plot(epochs, accuracy, label='Точность обучающей выборки')
  if val:
    plt.plot(epochs, val_accuracy, label='Точность тестовой выборки')
  plt.title('Точность классификации')
  plt.xlabel('число эпох')
  plt.legend()
  if fname:
    plt.savefig('./imgs/'+fname+'acc_opt.png')
  plt.show()