from ...config import ModelConfig
from ..layers import Encoder

import tensorflow as tf
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Model


class MTNet(Model):
    '''
    x: data input (batch_size, n, T, D)
    q: short term data (batch_size, T, D)
    n: length of x
    d: filters
    W: kernel size in time dimension
    T: seqence length
    D: variable dimension
    T_c = T - w + 1
    '''
    def __init__(self):
        super(MTNet, self).__init__(self)
        self.config = ModelConfig
        self.encoder = Encoder(config=ModelConfig)

    def call(self, x, q, training=False):
        # non-linear
        m_i = self.encoder(x)  # batch_size, T_c, d

        q = tf.reshape(q, shape=(-1, 1, *q.shape[1:]))
        u = self.encoder(q) # batch_size, T_c, d
        
        u_transposed = tf.transpose(u, perm=(0, 2, 1))  # batch_size, d, T_c
        # input memory representation
        _p_i = tf.matmul(m_i, u_transposed) # batch_size, T_c, T_c
        _p_i = _p_i[:, :, :1]
        # weights distribution vector
        p_i = tf.nn.softmax(_p_i, axis=1)
        
        c_i = self.encoder(x)   # batch_size, T_c, d
        # output memory representation
        o_i = tf.multiply(p_i, c_i) # batch_size, T_c, d

        _y_t_d = tf.concat([o_i, u], axis=-1)   # batch_size, T_c, 2*d
        _y_t_d = tf.reshape(_y_t_d, shape=(-1, tf.math.reduce_prod(_y_t_d.shape[1:], axis=0).numpy()))  # batch_size, T_c*2*d
        y_t_d = Dense(units=x.shape[-1])(_y_t_d)    # batch_size, D

        # autoregrssive
        assert self.config.W > 0
        y = q[:, 0, -(self.config.W-1):, :]    # batch_size, W-1, D
        _y_t_l = tf.reshape(y, shape=(-1, tf.math.reduce_prod(y.shape[1:], axis=0).numpy()))
        y_t_l = Dense(units=x.shape[-1])(_y_t_l)    # batch_size, D

        y_hat = y_t_d + y_t_l   # batch_size, D

        return y_hat
