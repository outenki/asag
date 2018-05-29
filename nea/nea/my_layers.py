# modified by outenki
# 2018.5.28
# Rewrite Attention layer

import keras.backend as K
from keras.engine.topology import Layer
from keras.layers.convolutional import Convolution1D
import keras.initializers as initializers
import ipdb

class Attention(Layer):
    def __init__(self, op='attsum', bias=True, activation='tanh', init_stdev=0.01, name='att', **kwargs):
        self.supports_masking = True
        assert op in {'attsum', 'attmean'}
        assert activation in {None, 'tanh'}
        self.op = op
        self.activation = activation
        self.init_stdev = init_stdev
        self.name = name
        self.att_weight = 0
        # self.init = initializers.get('glorot_normal')
        self.init = initializers.RandomNormal(stddev=self.init_stdev)
        self.bias = bias
        super(Attention, self).__init__(**kwargs)

    def build(self, input_shape):
        self.att_W = self.add_weight((input_shape[2], input_shape[2]),
                initializer = self.init,
                name='{}_att_W'.format(self.name))
        self.att_v = self.add_weight((input_shape[2],),
                initializer = self.init,
                name='{}_att_v'.format(self.name))
        if self.bias:
            self.att_b = self.add_weight((input_shape[2], ),
                    initializer = self.init,
                    name = '{}_att_b'.format(self.name))
        super(Attention, self).build(input_shape)

    def call(self, x, mask=None):
        # ipdb.set_trace()
        u = K.dot(x, self.att_W)
        if self.bias:
            u += self.att_b
        if self.activation == 'tanh':
            u = K.tanh(u)
        weights = K.dot(u, self.att_v)
        weights = K.exp(weights)
        # weights = K.softmax(weights)
        if mask:
            weights *= K.cast(mask, K.floatx())
        weights /= K.cast(K.sum(weights, axis=1, keepdims=True) + K.epsilon(), K.floatx())

        out = x * K.expand_dims(weights)
        if self.op == 'attsum':
            out = out.sum(axis=1)
        elif self.op == 'attmean':
            out = out.sum(axis=1) / mask.sum(axis=1, keepdims=True)
        return K.cast(out, K.floatx())

    def get_output_shape_for(self, input_shape):
        return (input_shape[0], input_shape[2])

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[2])
    
    def compute_mask(self, x, mask):
        return None
    
    def get_config(self):
        config = {'op': self.op, 'activation': self.activation, 'init_stdev': self.init_stdev}
        base_config = super(Attention, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

class MeanOverTime(Layer):
    def __init__(self, mask_zero=True, **kwargs):
        self.mask_zero = mask_zero
        self.supports_masking = True
        super(MeanOverTime, self).__init__(**kwargs)

    def call(self, x, mask=None):
        if self.mask_zero:
            return K.cast(x.sum(axis=1) / mask.sum(axis=1, keepdims=True), K.floatx())
            # return K.cast(x.reduce_sum(axis=1) / mask.reduce_sum(axis=1, keepdims=True), K.floatx())
        else:
            return K.mean(x, axis=1)

    def get_output_shape_for(self, input_shape):
        return (input_shape[0], input_shape[2])

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[2])
    
    def compute_mask(self, x, mask):
        return None
    
    def get_config(self):
        config = {'mask_zero': self.mask_zero}
        base_config = super(MeanOverTime, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

class Conv1DWithMasking(Convolution1D):
    def __init__(self, **kwargs):
        self.supports_masking = True
        super(Conv1DWithMasking, self).__init__(**kwargs)
    
    def compute_mask(self, x, mask):
        return mask
