import tensorflow as tf
from models.base_model import BaseModel
from math import floor

class cnnModel(BaseModel):
    def __init__(self, seq_length, nclasses) -> None:
        super().__init__(seq_length, nclasses)

    def conv1d_layer(self, 
                    _layer,
                    _filters = 32, 
                    _kernel_size = 4, 
                    _padding = 'same', 
                    _strides = 1,
                    _kernel_initializer = 'he_uniform',
                    _batch_norm = True,
                    _dropout = False,
                    _activation = 'relu'):
        # if self.nclasses < 3:
        #     _activation = 'sigmoid'
        self.layer = tf.keras.layers.Conv1D(filters = _filters, 
                        kernel_size = _kernel_size,  
                        padding = _padding, 
                        strides = _strides,
                        activation = _activation,
                        kernel_initializer = _kernel_initializer
                        )(_layer)
        if _batch_norm:
            self.layer = tf.keras.layers.BatchNormalization()(self.layer)
        if _dropout:
            self.layer = tf.keras.layers.Dropout()(self.layer)
        return self.layer

    def model_generator(self, dataset, params):
        self.layer=None
        self.model=None
        hidden_layers, solver, learning_rate, lr_decay, callback, epochs, kernel_size, activation = self.convert_params(params)
        self.format_params(params)
        self.conv1d_layer(self.input_layer, 
            _filters=hidden_layers[0], 
            _kernel_size=kernel_size if hidden_layers[0] > kernel_size else 4,
            _activation=activation)
        for i in range(len(hidden_layers)):
            self.conv1d_layer(self.layer, 
                _filters=hidden_layers[i], 
                _kernel_size=kernel_size if hidden_layers[i] > kernel_size else 4,
                _activation=activation)
        self.add_pooling()
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
        return self.result[1]


    def convert_params(self, params):
        hidden_layers = [params[0]]
        for i in range(1,5):
            if params[i] > 0:
                hidden_layers.append(params[i]) 
            else:
                break 
        hidden_layers = [round(num) for num in hidden_layers]       
        hidden_layers.sort(reverse=True) 
        print("-------------",hidden_layers)
        solver = [tf.keras.optimizers.Adam, tf.keras.optimizers.RMSprop, tf.keras.optimizers.SGD][floor(params[5])]
        learning_rate = 10**(-floor(params[6]))
        lr_decay = [True, False][round(params[7])]
        callback = [tf.keras.callbacks.EarlyStopping(monitor='loss', patience=5), None][round(params[8])]
        epochs = round(params[9])
        kernel_size = round(params[10])
        activation = ['relu', 'tanh'][round(params[11])]

        return hidden_layers, solver, learning_rate, lr_decay, callback, epochs, kernel_size, activation


    def format_params(self, params):
        hidden_layers, solver, learning_rate, lr_decay, callback, epochs, kernel_size, activation = self.convert_params(params)
        if callback:
            callback = True
        return "'hidden_layer_sizes'={}\n " \
            "'solver'='{}'\n " \
            "'learning_rate'='{}'\n " \
            "'lr_decay'={}\n " \
            "'callback'='{}'"\
            "'epochs'='{}'"\
            "'kernel_size'='{}'"\
            "'activation'='{}'"\
            .format(hidden_layers, solver, learning_rate, lr_decay, callback, epochs, kernel_size, activation)

'''
boundaries for hyperparameters:
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
kernel_size: [2 to 32]
activation: [relu, tanh] as [0,1]
'''