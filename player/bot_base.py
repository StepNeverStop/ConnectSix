import os
import time
import tensorflow as tf
from abc import ABC, abstractmethod

class Bot(ABC):
    """ A bot class. Override this class to implement your own Connect6 AI. """

    def __init__(self, dim = 19, name='bot'):
        assert type(dim) == int and dim > 0
        self.name=name
        self.dim = dim

    @abstractmethod
    def choose_action(self, state):
        '''
        必须实现这个方法，每个机器人或者选手必须有选择动作的函数
        '''
        raise NotImplementedError("Implement this to build your own AI.")
        pass

    def save_checkpoint(self, *args, **kargs):
        pass
    def restore(self, *args, **kargs):
        pass
    def writer_summary(self, *args, **kargs):
        pass
    def writer_loop_summary(self, *args, **kargs):
        pass
    def learn(self, *args, **kargs):
        pass
    def store(self, *args, **kargs):
        pass
    
class RL_Policy(Bot):
    def __init__(self, dim=19, name='rl_policy'):
        super().__init__(dim, name)
        tf.reset_default_graph()
        self.graph = tf.Graph()
        gpu_options = tf.GPUOptions(allow_growth=True)
        self.sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options), graph=self.graph)
        with self.graph.as_default():
            self.global_step = tf.get_variable('global_step', shape=(), initializer=tf.constant_initializer(value=0), trainable=False)
            self.saver = tf.train.Saver(max_to_keep=5, pad_step_number=True)
            self.writer = tf.summary.FileWriter(time.strftime("logs/%Y%m%d%H%M%S", time.localtime()), graph=self.graph)

    def save_checkpoint(self, global_step):
        self.saver.save(self.sess, os.path.join('./model', 'rb'), global_step=global_step, write_meta_graph=False)

    def restore(self, cp_dir='./model'):
        try:
            self.recorder.saver.restore(self.sess, tf.train.latest_checkpoint(cp_dir))
        except Exception as e:
            print(e)
            print('restore failed.')

    def writer_summary(self, x, ys):
        self.writer.add_summary(tf.Summary(
            value=[
                tf.Summary.Value(tag=y['tag'], simple_value=y['value']) for y in ys
            ]), x)

    def writer_loop_summary(self, global_step, **kargs):
        self.writer_summary(
            x=global_step,
            ys=[{'tag': 'MAIN/' + key, 'value': kargs[key]} for key in kargs]
        )

    @abstractmethod
    def learn(self):
        pass
        
    def store(self, **kargs):
        pass

