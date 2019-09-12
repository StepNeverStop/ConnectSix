import numpy as np
import tensorflow as tf
from .bot_base import RL_Policy
from utils.replay_buffer import ExperienceReplay

initKernelAndBias = {
    'kernel_initializer': tf.random_normal_initializer(0.0, .1),
    'bias_initializer': tf.constant_initializer(0.1, dtype=tf.float32)
}


class MyPolicy(RL_Policy):
    """
    实现自己的智能体策略
    """

    def __init__(self, dim, name='wjs_policy'):
        super().__init__(dim, name)
        
        self.state_dim = dim * dim * 3
        self.gamma = 0.99
        self.lr = 0.0005
        self.data = ExperienceReplay(batch_size = 100, capacity=10000)

        with self.graph.as_default():
            self.pl_s = tf.placeholder(tf.float32, [None, self.state_dim], 'state')
            self.pl_r = tf.placeholder(tf.float32, [None, 1], 'reward')
            self.pl_s_ = tf.placeholder(tf.float32, [None, self.state_dim], 'next_state')
            self.pl_done = tf.placeholder(tf.float32, [None, 1], 'done')
            
            self.v = self.v_net('v', self.pl_s)
            self.action = tf.argmax(self.v)
            self.v_ = self.v_net('v', self.pl_s_)
            self.predict = tf.stop_gradient(self.pl_r + self.gamma * self.v_ * (1 - self.pl_done))

            self.v_loss = tf.reduce_mean(tf.squared_difference(self.v, self.predict))
            self.v_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='v')
            optimizer = tf.train.AdamOptimizer(self.lr)
            self.train_v = optimizer.minimize(self.v_loss, var_list=self.v_vars, global_step=self.global_step)

            tf.summary.scalar('LOSS/v_loss', self.v_loss)
            self.summaries = tf.summary.merge_all()

            self.sess.run(tf.global_variables_initializer())
    
    def update_offset(self, offset):
        assert type(offset) == int
        self.offset = offset

    def v_net(self, name, input_vector):
        with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
            l1 = tf.layers.dense(input_vector, 128, tf.nn.relu, **initKernelAndBias)
            l2 = tf.layers.dense(l1, 64, tf.nn.relu, **initKernelAndBias)
            l3 = tf.layers.dense(l2, 32, tf.nn.relu, **initKernelAndBias)
            v = tf.layers.dense(l3, 1, None, **initKernelAndBias)
            return v

    def store(self, **kargs):
        self.data.add(*kargs.values())

    def choose_action(self, state):
        indexs, all_states = self.get_all_available_actions(state)
        if np.random.rand() > 0.2:
            action = self.sess.run(self.action, feed_dict={
                self.pl_s: all_states
            })[0]
        else:
            action = np.random.randint(len(indexs))
        x, y = indexs[action] % self.dim, indexs[action] // self.dim
        return x, y

    def learn(self):
        try:
            s, r, s_, done = self.data.sample()
            summaries, _ = self.sess.run([self.summaries, self.train_v], feed_dict={
                self.pl_s: np.eye(3)[s].reshape(s.shape[0],-1),
                self.pl_r: r[:, np.newaxis],
                self.pl_s_: np.eye(3)[s_].reshape(s.shape[0],-1),
                self.pl_done: done[:, np.newaxis]
            })
            self.writer.add_summary(summaries, self.sess.run(self.global_step))
        except Exception as e:
            print(e)
            return

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
