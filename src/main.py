import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import random

from utils.config import get_ga_config, get_hp_config, get_ts_data
from utils.plot_loss import plot_loss_curves
from utils.confusion_matrix import make_confusion_matrix
from data_loader.loader import data_loader
from models import cnn_model, rnn_model

random.seed(42)

#Todo add number of units for the loop
def generate_cnn_model(model, ga_setup=None):
    model.conv1d_layer(model.input_layer)
    for i in range(3):
        model.conv1d_layer(model.layer)
    model.add_pooling()
    model.build_model()
    if model.nclasses < 3:
        model.compile_model(_loss=tf.keras.losses.BinaryCrossentropy())
    else:
        model.compile_model()
    ...

def generate_rnn_model(model, ga_setup=None):
    if ga_setup.layer_type == "LSTM":
        model.lstm_layer(model.input_layer)
        for i in range(2):
            model.lstm_layer(model.layer)
    if ga_setup.layer_type == "GRU":
        model.gru_layer(model.input_layer)
        for i in range(2):
            model.lstm_layer(model.layer)
    model.build_model()
    if model.nclasses < 3:
        model.compile_model(_loss=tf.keras.losses.BinaryCrossentropy())
    else:
        model.compile_model()
    ...


def main():
    # Loading configuration data
    ga_config = get_ga_config()
    hp_config = get_hp_config()
    ts_config = get_ts_data()
    # Loading training and test datasets
    datasets = [data_loader(i, ts_config) for i in range(len(ts_config['DATASET_NAMES']))]
    # Initializing CNN and RNN models for each dataset with input_layer and output_layer
    cnn_models = [cnn_model.cnnModel(
        ts_config['MAX_SEQUENCE_LENGTH_LIST'][i],
        ts_config['NUB_CLASSES_LIST'][i]
        ) for i in range(len(ts_config['DATASET_NAMES']))
        ]
    rnn_models = [rnn_model.rnnModel(
        ts_config['MAX_SEQUENCE_LENGTH_LIST'][i],
        ts_config['NUB_CLASSES_LIST'][i]
        ) for i in range(len(ts_config['DATASET_NAMES']))
        ]
    
    # Generating baseline hidden layers for each CNN and RNN models

    print([model.layer for model in cnn_models])
    cnn_models[1].conv1d_layer(cnn_models[1].input_layer)
    cnn_models[1].conv1d_layer(cnn_models[1].layer)
    cnn_models[1].conv1d_layer(cnn_models[1].layer)
    # cnn_models[1].conv1d_layer(cnn_models[1].layer)
    # cnn_models[1].conv1d_layer(cnn_models[1].layer)
    cnn_models[1].add_pooling()
    cnn_models[1].build_model()
    if ts_config["NUB_CLASSES_LIST"][1] < 3:
        print("binary")
        cnn_models[1].compile_model(_loss=tf.keras.losses.BinaryCrossentropy())
    else:
        cnn_models[1].compile_model()
    cnn_models[1].model_train(
        train_data = datasets[1]["X_train"], 
        train_labels = datasets[1]["Y_train"],
        test_data = datasets[1]["X_test"], 
        test_labels = datasets[1]["Y_test"]
        )
    cnn_models[1].model_test(
        test_data = datasets[1]["X_test"], 
        test_lables = datasets[1]["Y_test"]
        )
    # print(cnn_models[1].model.summary())
    # print(cnn_models[1].history.history)
    # plot_loss_curves(cnn_models[1].history)
    cnn_models[1].model_predict(test_data = datasets[1]["X_test"])
    print(cnn_models[1].prediction)
    print(datasets[1]["Y_test"])
    print(ts_config["NUB_CLASSES_LIST"][1])
    make_confusion_matrix(datasets[1]["Y_test"], cnn_models[1].prediction)
    print(cnn_models[1].result[1])

    # rnn_models[1].lstm_layer(rnn_models[1].input_layer)
    # # rnn_models[1].layer=tf.expand_dims(rnn_models[1].layer, axis=-1)
    # # rnn_models[1].lstm_layer(rnn_models[1].layer, _return_sequences=True)
    # # rnn_models[1].lstm_layer(rnn_models[1].layer, _return_sequences=True)
    # # rnn_models[1].lstm_layer(rnn_models[1].layer, _return_sequences=True)
    # # rnn_models[1].lstm_layer(rnn_models[1].layer)
    # rnn_models[1].build_model()
    # rnn_models[1].model.summary()
    # if ts_config["NUB_CLASSES_LIST"][1] < 3:
    #     print("binary")
    #     rnn_models[1].compile_model(_loss=tf.keras.losses.BinaryCrossentropy())
    # else:
    #     rnn_models[1].compile_model()
    # rnn_models[1].model_train(
    #     train_data = datasets[1]["X_train"], 
    #     train_labels = datasets[1]["Y_train"],
    #     test_data = datasets[1]["X_test"], 
    #     test_labels = datasets[1]["Y_test"]
    #     )
    # rnn_models[1].model_test(
    #     test_data = datasets[1]["X_test"], 
    #     test_lables = datasets[1]["Y_test"]
    #     )
    # # print(rnn_models[1].model.summary())
    # # print(rnn_models[1].history.history)
    # # plot_loss_curves(rnn_models[1].history)
    # rnn_models[1].model_predict(test_data = datasets[1]["X_test"])
    # print(rnn_models[1].prediction)
    # print(datasets[1]["Y_test"])
    # print(ts_config["NUB_CLASSES_LIST"][1])
    # make_confusion_matrix(datasets[1]["Y_test"], rnn_models[1].prediction)

    # # print(datasets[1]['X_train'][1])
    # # print(tf.signal.fft(datasets[1]['X_train'][3]))
    # plt.plot(tf.signal.rfft(datasets[1]['X_train'][1]), label='fft')
    # # plt.plot(datasets[1]['X_train'][1], label='raw data')
    # # plt.yscale('log')
    # plt.xlabel('Frequency bin')
    # plt.ylabel('Difference')
    # plt.legend()
    # plt.show()
    

if __name__ == "__main__":
    main()