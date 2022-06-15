import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import random


from utils.config import get_ga_config, get_hp_config, get_ts_data
from utils.plot_loss import plot_loss_curves
from utils.confusion_matrix import make_confusion_matrix
from utils.eda import class_distribution, train_test_distribution
from data_loader.loader import data_loader

from models import cnn_model, rnn_model
from evolutionary_search import genetic_optimization

random.seed(42)

#Todo add number of units for the loop
def generate_cnn_model(model, dataset, ga_setup=None):
    model.conv1d_layer(model.input_layer)
    for i in range(3):
        model.conv1d_layer(model.layer)
    model.add_pooling()
    model.build_model()
    if model.nclasses < 3:
        model.compile_model(_loss=tf.keras.losses.BinaryCrossentropy())
    else:
        model.compile_model()
    model.model_train(
        train_data = dataset["X_train"], 
        train_labels = dataset["Y_train"],
        test_data = dataset["X_test"], 
        test_labels = dataset["Y_test"]
    )
    model.model_test(
        test_data = dataset["X_test"], 
        test_lables = dataset["Y_test"]
        )
    # print(model.model.summary())
    # print(model.history.history)
    if model.nclasses < 3:
        plot_loss_curves(model.history, binary=True)
    else:
        plot_loss_curves(model.history)

    model.model_predict(test_data = dataset["X_test"])
    # print(model.prediction)
    # print(dataset["Y_test"])
    # print(ts_config["NUB_CLASSES_LIST"][6])
    make_confusion_matrix(dataset["Y_test"], model.prediction)
    print(model.result[1])

def generate_rnn_model(model, dataset, ga_setup=None):
    # if ga_setup.layer_type == "LSTM":
    #     model.lstm_layer(model.input_layer)
    #     for i in range(1):
    #         model.lstm_layer(model.layer)
    # if ga_setup.layer_type == "GRU":
    #     model.gru_layer(model.input_layer)
    #     for i in range(1):
    #         model.lstm_layer(model.layer)
    model.lstm_layer(model.input_layer, _units=32, _return_sequences=True)
    model.lstm_layer(model.layer, _units=32)
    # model.gru_layer(model.input_layer, _units=32, _return_sequences=True)
    # model.gru_layer(model.input_layer, _units=32)
    # for i in range(1):
    #     model.lstm_layer(model.layer)
    model.build_model()
    if model.nclasses < 3:
        model.compile_model(_loss=tf.keras.losses.BinaryCrossentropy())
    else:
        model.compile_model()
        model.model_train(
        train_data = dataset["X_train"], 
        train_labels = dataset["Y_train"],
        test_data = dataset["X_test"], 
        test_labels = dataset["Y_test"],
        # _callbacks = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=5)
    )
    model.model_test(
        test_data = dataset["X_test"], 
        test_lables = dataset["Y_test"]
        )

    if model.nclasses < 3:
        plot_loss_curves(model.history, binary=True)
    else:
        plot_loss_curves(model.history)

    model.model_predict(test_data = dataset["X_test"])
    # print(model.prediction)
    # print(dataset["Y_test"])
    make_confusion_matrix(dataset["Y_test"], model.prediction)
    print(model.result[1])

def solver(model, dataset, hp_range):
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

    # EDA 
    # for dataset in datasets:
    #     class_distribution(dataset)
    #     train_test_distribution(dataset)

    # Generating baseline CNN and RNN models
    generate_cnn_model(cnn_models[2], datasets[2])
    # generate_rnn_model(rnn_models[2], datasets[2])

    #Initializing GA solver



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