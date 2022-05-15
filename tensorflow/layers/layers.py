from ...config.config import ModelConfig

import tensorflow as tf
from tensorflow.keras.layers import Layer, Dense, GRU, Conv2D


class BahdanauAttention(Layer):
  def __init__(self, units, **kwargs):
    super(BahdanauAttention, self).__init__(**kwargs)
    self.W1 = Dense(units)
    self.W2 = Dense(units)
    self.V = Dense(1)

  def call(self, query, values):
    '''
    query: previous output (batch_size, hidden size)
    values: last hidden states of each time in the encoders (batch_size, T, hidden_size)
    '''    
    query_with_time_axis = tf.expand_dims(query, 1) # (batch_size, 1, hidden_size)
    
    _score = tf.nn.tanh(self.W1(query_with_time_axis) + self.W2(values)) # (batch_size, T, units)
    score = self.V(_score)   # (batch_size, T, 1)
    
    attention_weights = tf.nn.softmax(score, axis=1)    #(batch_size, T, 1)

    context_vector = attention_weights * values    #(batch_size, T, 1)
    context_vector = tf.reduce_sum(context_vector, axis=1)  # (batch_size, hidden_size)

    return context_vector, attention_weights


class Encoder(Layer):
    def __init__(
        self, 
        num_filters: int, 
        config=ModelConfig, 
        **kwargs):
        super(Encoder, self).__init__(self, **kwargs)
        self.config = config
        
    def call(self, x):
        '''
        x: data input (batch_size, n, T, D)
        n: length of x
        W: kernel size in time dimension
        T: seqence length
        D: variable dimension
        T_c = T - w + 1
        '''
        x_reshaped = tf.reshape(x, shape=(-1, self.coffig.T, self.config.D, 1)) # batch_size * n, T, D ,1
        
        # CNN layer
        conv_output = Conv2D(
            filters=self.config.num_filters, 
            kernel_size=(self.config.W, x.shape[-1]),
            padding=self.config.padding,
            activation=self.config.activation
        )(x_reshaped)  # T_c x 1 x d_c
        conv_output = Dropout(rate=self.config.rate)(conv_output)
        conv_output = tf.squeeze(conv_output, axis=2)   # T_c x d_c

