import tensorflow as tf
from models.base_model import BaseModel


class cnnModel(BaseModel):
    def __init__(self, seq_length, nclasses) -> None:
        super().__init__(seq_length, nclasses)

    def conv1d_layer(self, 
                    _layer,
                    _filters = 32, 
                    _kernel_size = 16, 
                    _padding = 'same', 
                    _strides = 1,
                    _kernel_initializer = 'he_uniform',
                    _batch_norm = True,
                    _dropout = False,
                    _activation = 'relu'):
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
