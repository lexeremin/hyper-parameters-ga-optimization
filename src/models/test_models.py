from tensorflow import keras
# import numpy as np

def cnn_model(MAX_SEQUENCE_LENGTH=1000, NB_CLASS=1, NUM_CELLS=8):

    input_layer = keras.layers.Input(shape=(MAX_SEQUENCE_LENGTH, 1))
    #He weight initialization is W=np.random.rand(shape)*np.sqrt(2/n^[l-1])
    conv1 = keras.layers.Conv1D(filters=128, kernel_size=3, padding="same", kernel_initializer='he_uniform')(input_layer)
    conv1 = keras.layers.BatchNormalization()(conv1)
    # conv1 = keras.layers.Dropout()(conv1)
    conv1 = keras.layers.ReLU()(conv1)

    conv2 = keras.layers.Conv1D(filters=256, kernel_size=5, padding="same", kernel_initializer='he_uniform')(conv1)
    conv2 = keras.layers.BatchNormalization()(conv2)
    # conv2 = keras.layers.Dropout()(conv1)
    conv2 = keras.layers.ReLU()(conv2)

    conv3 = keras.layers.Conv1D(filters=128, kernel_size=3, padding="same", kernel_initializer='he_uniform')(conv2)
    conv3 = keras.layers.BatchNormalization()(conv3)
    # conv3 = keras.layers.Dropout()(conv1)
    conv3 = keras.layers.ReLU()(conv3)

    gap = keras.layers.GlobalAveragePooling1D()(conv3)

    output_layer = keras.layers.Dense(NB_CLASS, activation="softmax")(gap)

    return keras.models.Model(inputs=input_layer, outputs=output_layer)


def rnn_model(MAX_SEQUENCE_LENGTH=1000, NB_CLASS=1, NUM_CELLS=8):

    input_layer = keras.layers.Input(shape=(MAX_SEQUENCE_LENGTH,1))

    lstm = keras.layers.LSTM(
            128, #number of units
            activation = 'tanh', 
            kernel_initializer = 'glorot_uniform',
            )(input_layer)
    # lstm = keras.layers.Dropout()(lstm)
    # gap = keras.layers.GlobalAveragePooling1D()(lstm)

    output_layer = keras.layers.Dense(NB_CLASS, activation="softmax")(lstm)

    return keras.models.Model(inputs=input_layer, outputs=output_layer)