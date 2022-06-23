import tensorflow as tf
from models.base_model import BaseModel
from math import floor
class rnnModel(BaseModel):
    def __init__(self, seq_length, nclasses) -> None:
        super().__init__(seq_length, nclasses)    
    
    def lstm_layer(self, 
                    _layer,
                    _units = 32, 
                    _kernel_initializer = 'glorot_uniform',
                    _batch_norm = True,
                    _dropout = 0.0,
                    _recurrent_dropout = 0.0,
                    _activation = 'tanh',
                    _recurrent_activation = "sigmoid",
                    _return_sequences=False):
        # if self.nclasses < 3:
        #     _activation = 'sigmoid'
        self.layer = tf.keras.layers.LSTM(_units,
                        activation = _activation,
                        dropout = _dropout,
                        recurrent_activation = _recurrent_activation,
                        kernel_initializer = _kernel_initializer,
                        recurrent_dropout = _recurrent_dropout,
                        return_sequences=_return_sequences
                        )(_layer)
        if _batch_norm:
            self.layer = tf.keras.layers.BatchNormalization()(self.layer)
        return self.layer

    def gru_layer(self, 
                    _layer,
                    _units = 32, 
                    _kernel_initializer = 'glorot_uniform',
                    _batch_norm = True,
                    _dropout = 0.0,
                    _recurrent_dropout = 0.0,
                    _activation = 'tanh',
                    _recurrent_activation = "sigmoid",
                    _return_sequences=False):
        # if self.nclasses < 3:
        #     _activation = 'sigmoid'
        self.layer = tf.keras.layers.GRU( _units,
                        activation = _activation,
                        dropout = _dropout,
                        recurrent_activation = _recurrent_activation,
                        kernel_initializer = _kernel_initializer,
                        recurrent_dropout = _recurrent_dropout,
                        return_sequences=_return_sequences
                        )(_layer)
        if _batch_norm:
            self.layer = tf.keras.layers.BatchNormalization()(self.layer)
        return self.layer

    def model_generator(self, dataset, params):
        self.layer=None
        self.output_layer = tf.keras.layers.Dense(self.nclasses, activation="softmax") if self.nclasses > 2 else tf.keras.layers.Dense(1, activation="sigmoid")
        hidden_layers, solver, learning_rate, lr_decay, callback, epochs, layer_type = self.convert_params(params)
        print(self.format_params(params))

        if layer_type == 'lstm':
            if len(hidden_layers) < 2:
                self.lstm_layer(self.input_layer, _units=hidden_layers[0])
            else:
                self.lstm_layer(self.input_layer, _units=hidden_layers[0], _return_sequences=True)
            for i in range(1,len(hidden_layers)):
                if i<len(hidden_layers)-1:
                    self.lstm_layer(self.layer, _units=hidden_layers[i], _return_sequences=True)
                else:
                    self.lstm_layer(self.layer, _units=hidden_layers[i])
        if layer_type == 'gru':
            if len(hidden_layers) < 2:
                self.gru_layer(self.input_layer, _units=hidden_layers[0])
            else:
                self.gru_layer(self.input_layer, _units=hidden_layers[0], _return_sequences=True)
            for i in range(1,len(hidden_layers)):
                if i<len(hidden_layers)-1:
                    self.gru_layer(self.layer, _units=hidden_layers[i], _return_sequences=True)
                else:
                    self.gru_layer(self.layer, _units=hidden_layers[i])
        # self.add_pooling()
        self.build_model()
        if self.nclasses < 3:
            self.compile_model(
                _loss=tf.keras.losses.BinaryCrossentropy(), 
                _optimizer=solver,
                _learning_rate=learning_rate,
                _lr_decay=lr_decay)
        else:
            self.compile_model(
                _optimizer=solver,
                _learning_rate=learning_rate,
                _lr_decay=lr_decay)
        self.model_train(
            train_data = dataset["X_train"], 
            train_labels = dataset["Y_train"],
            test_data = dataset["X_test"], 
            test_labels = dataset["Y_test"],
            _callbacks = callback,
            _epoches = epochs
        )
        self.model_test(
            test_data = dataset["X_test"], 
            test_lables = dataset["Y_test"]
        )
        self.model_predict(test_data = dataset["X_test"])
        print('accuracy: ', self.result[1])
        print('----END')
        # self.model.summary()
        return self.result[1]


    def convert_params(self, params):
        hidden_layers = [params[0]]
        for i in range(1,5):
            if params[i] > 0:
                hidden_layers.append(params[i]) 
            else:
                break 
        hidden_layers = [round(num) for num in hidden_layers ] 
        hidden_layers.sort(reverse=True)
        for i in range(len(hidden_layers)):
            if hidden_layers[i] <=0:
                hidden_layers.pop(i)
        solver = [tf.keras.optimizers.Adam, tf.keras.optimizers.RMSprop, tf.keras.optimizers.SGD][floor(params[5])]
        learning_rate = 10**(-floor(params[6]))
        lr_decay = [True, False][round(params[7])]
        callback = [tf.keras.callbacks.EarlyStopping(monitor='loss', patience=5), None][round(params[8])]
        epochs = round(params[9])
        layer_type = ['lstm', 'gru'][round(params[10])]

        return hidden_layers, solver, learning_rate, lr_decay, callback, epochs, layer_type

    def format_params(self, params):
        hidden_layers, solver, learning_rate, lr_decay, callback, epochs, layer_type = self.convert_params(params)
        if callback:
            callback = True
        return "'hidden_layer_sizes'={}\n " \
            "'solver'={}\n " \
            "'learning_rate'={}\n " \
            "'lr_decay'={}\n" \
            "'callback'={}\n"\
            "'epochs'={}\n"\
            "'layer_type'={}"\
            .format(hidden_layers, solver, learning_rate, lr_decay, callback, epochs, layer_type)

'''
Reminder - boundaries for hyperparameters:
----
layer 1: [8 to 128]
layer 2:  [8 to 128]
layer 3: [8 to 128]
layer 4: [8 to 64]
layer 5: [8 to 32]
solver: [Adam, RMSProps, SGD] as [0, 1, 2]
learning_rate: [1-e6 to 1-e1] as [1,2,3,4,5,6]
lr_decay: [constant, exponential] as [0,1]
callback: [true, false]
epochs: [20 to 50]
layer_type: [lstm, gru] as [0,1]
'''
   