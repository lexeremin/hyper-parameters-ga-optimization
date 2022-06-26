import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import random
import argparse


from utils.config import get_ga_config, get_hp_config, get_ts_data
from utils.plot_loss import plot_loss_curves
from utils.confusion_matrix import make_confusion_matrix
from utils.eda import class_distribution, train_test_distribution
from data_loader.loader import data_loader

from models import cnn_model, rnn_model
from evolutionary_search import genetic_optimization

# random.seed(42)
tf.random.set_seed(42)

# test CNN model


def generate_cnn_model(model, dataset, params=None):
    model.conv1d_layer(model.input_layer)
    for i in range(3):
        model.conv1d_layer(model.layer)
    model.add_pooling()
    model.build_model()
    if model.nclasses < 3:
        model.compile_model(_loss=tf.keras.losses.BinaryCrossentropy(),
                            _learning_rate=0.001,
                            _lr_decay=True,
                            )
    else:
        model.compile_model(
            _lr_decay=True
        )
    model.model_train(
        train_data=dataset["X_train"],
        train_labels=dataset["Y_train"],
        test_data=dataset["X_test"],
        test_labels=dataset["Y_test"],
        _callbacks=tf.keras.callbacks.EarlyStopping(monitor='loss', patience=5)
    )
    model.model_test(
        test_data=dataset["X_test"],
        test_lables=dataset["Y_test"]
    )
    if model.nclasses < 3:
        plot_loss_curves(model.history, binary=True)
    else:
        plot_loss_curves(model.history)

    model.model_predict(test_data=dataset["X_test"])
    make_confusion_matrix(dataset["Y_test"], model.prediction)
    print(model.result[1])

    return model.result[1]

# test RNN model


def generate_rnn_model(model, dataset, params=None):
    model.lstm_layer(model.input_layer, _units=128)
    # model.lstm_layer(model.layer, _units=128)
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
        train_data=dataset["X_train"],
        train_labels=dataset["Y_train"],
        test_data=dataset["X_test"],
        test_labels=dataset["Y_test"],
        # _callbacks = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=5)
    )
    model.model_test(
        test_data=dataset["X_test"],
        test_lables=dataset["Y_test"]
    )

    if model.nclasses < 3:
        plot_loss_curves(model.history, binary=True)
    else:
        plot_loss_curves(model.history)

    model.model_predict(test_data=dataset["X_test"])
    # print(model.prediction)
    # print(dataset["Y_test"])
    make_confusion_matrix(dataset["Y_test"], model.prediction)
    print(model.result[1])


def main():
    # ---Loading configuration data
    ga_config = get_ga_config()
    ts_config = get_ts_data()

    # ---Loading training and test datasets and normalizing them
    datasets = [data_loader(i, ts_config)
                for i in range(len(ts_config['DATASET_NAMES']))]

    # ---Initializing CNN or RNN models for each dataset with input_layer and output_layer
    if ga_config['NN_TYPE'] == 'CNN':
        models = [cnn_model.cnnModel(
            ts_config['MAX_SEQUENCE_LENGTH_LIST'][i],
            ts_config['NUB_CLASSES_LIST'][i]
        ) for i in range(len(ts_config['DATASET_NAMES']))
        ]
    if ga_config['NN_TYPE'] == 'RNN':
        models = [rnn_model.rnnModel(
            ts_config['MAX_SEQUENCE_LENGTH_LIST'][i],
            ts_config['NUB_CLASSES_LIST'][i]
        ) for i in range(len(ts_config['DATASET_NAMES']))
        ]

    # ---EDA for timeseries datasets
    # for i in ga_config['DATASETS']:
    #     class_distribution(datasets[i], ts_config['DATASET_NAMES'][i])
    #     train_test_distribution(datasets[i], ts_config['DATASET_NAMES'][i])

    # ---Automated optimization for all datasets
    for i in ga_config['DATASETS']:
        fname = ga_config['NN_TYPE']+ts_config['DATASET_NAMES'][i]
        ga = genetic_optimization.GeneticSearch(
            ga_config, models[i], datasets[i])
        ga.initial_setup()
        ga.create_population()
        ga.define_operators()
        ga.solver()
        make_confusion_matrix(
            ga.dataset["Y_test"], ga.model.prediction, fname=fname)
        if ga.model.nclasses < 3:
            plot_loss_curves(ga.model.history, binary=True, fname=fname)
        else:
            plot_loss_curves(ga.model.history, fname=fname)

    # ---Generating baseline CNN and RNN models
    # generate_cnn_model(models[3], datasets[3])
    # generate_rnn_model(models[1], datasets[1])

    # ---Specific dataset optimization
    # # fname = NN_TYPE+ts_config['DATASET_NAMES'][i]
    # ga = genetic_optimization.GeneticSearch(ga_config, models[3], datasets[3])
    # ga.initial_setup()
    # ga.create_population()
    # ga.define_operators()
    # ga.solver()
    # make_confusion_matrix(ga.dataset["Y_test"], ga.model.prediction)
    # if ga.model.nclasses < 3:
    #     plot_loss_curves(ga.model.history, binary=True)
    # else:
    #     plot_loss_curves(ga.model.history)

    # ---Time series analysis
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
