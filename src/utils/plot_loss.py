import matplotlib.pyplot as plt

# Plot the validation and training data separately


def plot_loss_curves(history, val=True, binary=False, fname=False):

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
    plt.plot(epochs, loss, label='Training data')
    if val:
        plt.plot(epochs, val_loss, label='Test data')
    plt.title('Loss fuction')
    plt.xlabel('Epochs')
    plt.legend()
    # Plot accuracy
    if fname:
        plt.savefig('./imgs/'+fname+'loss_opt.png')
    plt.figure()
    plt.plot(epochs, accuracy, label='Training data')
    if val:
        plt.plot(epochs, val_accuracy, label='Test data')
    plt.title('Classification accuracy')
    plt.xlabel('Epochs')
    plt.legend()
    if fname:
        plt.savefig('./imgs/'+fname+'acc_opt.png')
    plt.show()
