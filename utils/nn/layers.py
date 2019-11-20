import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Conv2D

activation_fn = tf.keras.activations.tanh


class mlp(Sequential):
    def __init__(self, hidden_units, act_fn=activation_fn, output_shape=1, out_activation=None, out_layer=True):
        """
        Args:
            hidden_units: like [32, 32]
            output_shape: units of last layer
            out_activation: activation function of last layer
            out_layer: whether need specifing last layer or not
        """
        super().__init__()
        for u in hidden_units:
            self.add(Dense(u, act_fn))
        if out_layer:
            self.add(Dense(output_shape, out_activation))


class convs(Sequential):

    def __init__(self, act_fn=activation_fn, filters=[32, 64, 128],
                 kernel_size=[[3, 3],
                              [3, 3],
                              [3, 3]],
                 padding="same", data_format="channels_last"):
        super().__init__()
        for i in range(len(filters)):
            self.add(Conv2D(filters=filters[i], kernel_size=kernel_size[i], padding=padding, data_format=data_format, activation=act_fn))
