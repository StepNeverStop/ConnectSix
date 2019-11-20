import tensorflow as tf
from utils.nn.layers import mlp, convs
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Conv2D, Flatten


class V(tf.keras.Model):
    '''
    use for evaluate the value given a state.
    input: vector of state
    output: v(s)
    '''

    def __init__(self, vector_dim, name, hidden_units=[128, 64, 32]):
        super().__init__(name=name)
        self.net = mlp(hidden_units, output_shape=1, out_activation=None)
        self(tf.keras.Input(shape=vector_dim))

    def call(self, vector_input):
        v = self.net(vector_input)
        return v


class PV(tf.keras.Model):

    def __init__(self, dim, name):
        super().__init__(name=name)
        self.visual_net = convs(
            filters=[32, 64, 128],
            kernel_size=[[3, 3]
                         [3, 3],
                         [3, 3]],
            padding="same",
            data_format="channels_last"
        )
        self.action_conv = Sequential(
            Conv2D(
                filters=8,
                kernel_size=[1, 1],
                padding="same",
                data_format="channels_last",
                activation="tanh"
            ),
            Flatten()
        )
        self.action_dense = mlp(
            hidden_units=[512, 512],
            output_shape=dim[0] ** 2,
            out_activation=tf.nn.log_softmax,
            out_layer=True
        )
        self.v_conv = Sequential(
            Conv2D(
                filters=4,
                kernel_size=[2, 2],
                padding="same",
                data_format="channels_last",
                activation="tanh"
            ),
            Flatten()
        )
        self.v_dense = mlp(
            hidden_units=[256],
            output_shape=1,
            out_activation="tanh",
            out_layer=True
        )
        self(tf.keras.Input(shape=dim))

    def call(self, x):
        x = self.visual_net(x)
        actions_prob = self.action_dense(
            self.action_conv(x)
        )
        v = self.v_dense(
            self.v_conv(x)
        )
        return actions_prob, v
