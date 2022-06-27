from gc import callbacks
import tensorflow as tf
import numpy as np
import datetime

class BaseModel(tf.keras.Model):
    def __init__(self, seq_length, nclasses, name) -> None:
        super(BaseModel, self).__init__()
        self.nclasses = nclasses
        self.seq_length = seq_length
        self.input_layer = tf.keras.layers.Input(shape=(seq_length, 1))
        self.layer = None
        self.model = None
        self.output_layer = tf.keras.layers.Dense(
            nclasses, activation="softmax") if nclasses > 2 else tf.keras.layers.Dense(1, activation="sigmoid")
        self.history = None
        self.result = None
        self.prediction = None

        self.experiment_name = name

    def add_pooling(self) -> None:
        self.layer = tf.keras.layers.GlobalAveragePooling1D()(self.layer)

    def build_model(self) -> None:
        self.model = tf.keras.models.Model(
            inputs=self.input_layer, outputs=self.output_layer(self.layer))

    def compile_model(self, _loss=tf.keras.losses.SparseCategoricalCrossentropy(),
                      _optimizer=tf.keras.optimizers.Adam,
                      _learning_rate=0.001,
                      _lr_decay=False) -> None:

        if _lr_decay:
            _learning_rate = tf.keras.optimizers.schedules.ExponentialDecay(
                initial_learning_rate=_learning_rate,
                decay_steps=10000,
                decay_rate=0.9
            )
        if self.nclasses < 3:
            _loss = tf.keras.losses.BinaryCrossentropy()
            _metrics = tf.keras.metrics.BinaryAccuracy()
        else:
            _metrics = tf.keras.metrics.SparseCategoricalAccuracy()
        self.model.compile(
            loss=_loss,
            optimizer=_optimizer(_learning_rate),
            metrics=[_metrics]
        )

    def model_train(self, train_data, train_labels, test_data, test_labels, _epoches=30, _callback=None) -> None:
        _callbacks = []
        if _callback == None:
            _callbacks.append(self.create_tensorboard_callback('experiments'))
        else:
            _callbacks.append(self.create_tensorboard_callback('experiments'))
            _callbacks.append(_callback)
        self.history = self.model.fit(
            train_data,
            train_labels,
            epochs=_epoches,
            validation_data=(test_data, test_labels),
            callbacks=_callbacks
        )

    def model_test(self, test_data, test_lables) -> None:
        self.result = self.model.evaluate(test_data, test_lables)
        print(type(self.result))

    def model_predict(self, test_data) -> None:
        if self.nclasses > 2:
            self.prediction = np.argmax(self.model.predict(test_data), axis=1)
        else:
            self.prediction = self.model.predict(test_data).round()

    def model_accuracy(self):
        if self.result:
            return self.result[1]
        else:
            return None

    def create_tensorboard_callback(self, dir_name):

        log_dir = dir_name + "/" + self.experiment_name + "/" + \
            datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        tensorboard_callback = tf.keras.callbacks.TensorBoard(
            log_dir=log_dir
        )
        print(f"Saving TensorBoard log files to: {log_dir}")
        return tensorboard_callback
