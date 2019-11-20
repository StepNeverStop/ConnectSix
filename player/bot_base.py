import os
import time
import tensorflow as tf
from abc import ABC, abstractmethod


class Bot(ABC):
    """ A bot class. Override this class to implement your own Connect6 AI. """

    def __init__(self, dim=19, name='bot'):
        assert isinstance(dim, int) and dim > 0
        self.name = name
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


class RL_Policy(tf.keras.Model):
    def __init__(self, **kwargs):
        super().__init__()
        physical_devices = tf.config.experimental.list_physical_devices('GPU')
        if len(physical_devices) > 0:
            self.device = "/gpu:0"
            tf.config.experimental.set_memory_growth(physical_devices[0], True)
        else:
            self.device = "/cpu:0"
        tf.keras.backend.set_floatx('float32')
        self.global_step = tf.Variable(0, name="global_step", trainable=False, dtype=tf.int64)
        self.checkpoint = tf.train.Checkpoint(policy=self)
        self.saver = tf.train.CheckpointManager(self.checkpoint, directory='./model', max_to_keep=5, checkpoint_name='rb')
        self.writer = tf.summary.create_file_writer(time.strftime("logs/%Y%m%d%H%M%S", time.localtime()))

    def save_checkpoint(self, global_step):
        self.saver.save(checkpoint_number=global_step)

    def restore(self, cp_dir='./model'):
        if os.path.exists(os.path.join(cp_dir, 'checkpoint')):
            try:
                self.checkpoint.restore(self.saver.latest_checkpoint)
            except:
                print('restore model from checkpoint FAILED.')
            else:
                print('restore model from checkpoint SUCCUESS.')
        else:
            raise Exception(f'model file {cp_dir} cannot be found')

    def writer_loop_summary(self, global_step, **kargs):
        self.writer.set_as_default()
        tf.summary.experimental.set_step(global_step)
        for i in [{'tag': 'MAIN/' + key, 'value': kargs[key]} for key in kargs]:
            tf.summary.scalar(i['tag'], i['value'])
        self.writer.flush()

    @abstractmethod
    def learn(self):
        pass

    def store(self, **kargs):
        pass

    def write_training_summaries(self, summaries: dict):
        '''
        write tf summaries showing in tensorboard.
        '''
        for key, value in summaries.items():
            tf.summary.scalar(key, value)
