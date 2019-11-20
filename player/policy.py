import numpy as np
import tensorflow as tf
from .bot_base import RL_Policy
from utils.replay_buffer import ExperienceReplay
from utils.nn.nets import V


class MyPolicy(RL_Policy):
    """
    实现自己的智能体策略
    """

    def __init__(self, dim, name='wjs_policy'):
        super().__init__(dim, name)

        self.state_dim = dim * dim * 3
        self.gamma = 0.99
        self.lr = 0.0005
        self.data = ExperienceReplay(batch_size=100, capacity=10000)
        self.v_net = V(vector_dim=self.state_dim, name='v_net', hidden_units=[128, 64, 32])
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=self.lr)

    def update_offset(self, offset):
        assert isinstance(offset, int)
        self.offset = offset

    def store(self, **kargs):
        self.data.add(*kargs.values())

    @tf.function
    def _get_action(self, state):
        return tf.argmax(self.v_net(state))

    def choose_action(self, state):
        indexs, all_states = self.get_all_available_actions(state)
        if np.random.rand() > 0.2:
            action = self._get_action(all_states)[0]
        else:
            action = np.random.randint(len(indexs))
        x, y = indexs[action] % self.dim, indexs[action] // self.dim
        return x, y

    def learn(self):
        try:
            s, r, s_, done = self.data.sample()
            s = np.eye(3)[s].reshape(s.shape[0], -1)
            r = r[:, np.newaxis]
            s_ = np.eye(3)[s_].reshape(s.shape[0], -1)
            done = done[:, np.newaxis]
            summaries = self.train(s, r, s_, done)
            tf.summary.experimental.set_step(self.global_step)
            self.write_training_summaries(summaries)
            tf.summary.scalar('LEARNING_RATE/lr', self.lr)
            self.recorder.writer.flush()
        except Exception as e:
            print(e)
            return

    @tf.function
    def train(self, s, r, s_, done):
        with tf.device(self.device):
            with tf.GradientTape() as tape:
                v = self.v_net(s)
                v_ = self.v_net(s_)
                predict = tf.stop_gradient(r + self.gamma * v_ * (1 - done))
                v_loss = tf.reduce_mean((v - predict) ** 2)
            grads = tape.gradient(v_loss, self.v_net.trainable_variables)
            self.optimizer.apply_gradients(
                zip(grads, self.v_net.trainable_variables)
            )
            self.global_step.assign_add(1)
            return dict([
                ['LOSS/v_loss', v_loss]
            ])

    def get_all_available_actions(self, state):
        assert isinstance(state, np.ndarray), "state不是numpy类型"
        indexs = []
        for i in range(state.shape[0]):
            if state[i] == 2:
                indexs.append(i)
        all_states = []
        for i in indexs:
            a = np.zeros_like(state)
            a[i] = self.offset
            all_states.append(state - a)
        return indexs, np.array([np.eye(3)[i].reshape(-1) for i in all_states])
