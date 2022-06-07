import tensorflow as tf
from models.base_model import BaseModel

class rnnModel(BaseModel):
    def __init__(self, seq_length, nclasses) -> None:
        super().__init__(seq_length, nclasses)    
    
    def lstm_layer(self, 
                    _layer,
                    _units = 128, 
                    _kernel_initializer = 'glorot_uniform',
                    _batch_norm = True,
                    _dropout = 0.0,
                    _recurrent_dropout = 0.0,
                    _activation = 'tanh',
                    _recurrent_activation = "sigmoid",
                    _return_sequences=False):
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
                    _units = 256, 
                    _kernel_initializer = 'glorot_uniform',
                    _batch_norm = True,
                    _dropout = 0.0,
                    _recurrent_dropout = 0.0,
                    _activation = 'tanh',
                    _recurrent_activation = "sigmoid"):
        self.layer = tf.keras.layers.GRU( _units,
                        activation = _activation,
                        dropout = _dropout,
                        recurrent_activation = _recurrent_activation,
                        kernel_initializer = _kernel_initializer,
                        recurrent_dropout = _recurrent_dropout
                        )(_layer)
        if _batch_norm:
            self.layer = tf.keras.layers.BatchNormalization()(self.layer)
        return self.layer
   