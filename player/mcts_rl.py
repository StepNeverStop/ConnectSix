
import numpy as np
import copy
import tensorflow as tf
from .bot_base import RL_Policy
from utils.replay_buffer import ExperienceReplay
from utils.nn.nets import PV


class PV(RL_Policy):
    def __init__(self, dim, name='wjs_policy'):
        super().__init__(dim, name)
        self.lr = 0.0005
        self.data = ExperienceReplay(batch_size=100, capacity=10000)
        self.net = PV(dim=[self.dim, self.dim, 8], name='pv_net')
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=self.lr)

        @tf.function
        def _get_probs_and_v(self, state):
            return self.net(state)

        def get_probs_and_v(self, game):
            state = game.get_current_state()
            log_actions_prob, value = self._get_probs_and_v(state)
            actions_prob = np.exp(log_actions_prob)
            available_actions_prob = zip(game.available_actions, actions_prob[0][game.available_actions])
            return available_actions_prob, value

        def learn(self):
            s, p, v = self.data.sample()
            summaries = self.train(s, p, v)
            tf.summary.experimental.set_step(self.global_step)
            self.write_training_summaries(summaries)
            tf.summary.scalar('LEARNING_RATE/lr', self.lr)
            self.recorder.writer.flush()

        @tf.function
        def train(s, p, v):
            with tf.device(self.device):
                with tf.GradientTape() as tape:
                    action_probs, predict_v = self.net(s)
                    p_loss = -tf.reduce_mean(tf.reduce_sum(tf.multiply(p, action_probs), axis=-1))
                    v_loss = tf.reduce_mean((v - predict_v) ** 2)
                    loss = v_loss + p_loss
                grads = tape.gradient(loss, self.net.trainable_variables)
                self.optimizer.apply_gradients(
                    zip(grads, self.net.trainable_variables)
                )
                self.global_step.assign_add(1)
                return dict([
                    ['LOSS/v_loss', v_loss],
                    ['LOSS/p_loss', p_loss],
                    ['LOSS/loss', loss],
                ])

        def store(self, **kargs):
            self.data.add(*kargs.values())


class Node(object):

    def __init__(self, parent, prior_prob):
        '''
        parent: 父节点
        prior_prob: 选择该节点的先验概率
        '''
        self.parent = parent
        self.children = {}
        self.n = 0
        self.q = 0
        self.w = 0
        self.p = prior_prob

    def expand(self, available_actions_prob):
        '''
        树扩展
        '''
        for action, prob in available_actions_prob:
            if action not in self.children:
                self.children[action] = Node(self, prob)

    def select(self, c_puct):
        '''
        给当前节点选择一个动作， c_puct控制探索
        '''
        return max(self.children.items(), key=lambda item: item[-1].get_Q(c_puct))

    def update_recursive(self, leaf_value, player_step):
        '''
        递归函数，向上更新 n, w, q
        '''
        if self.parent:
            self.parent.update_recursive(leaf_value if player_step == 0 else -leaf_value, (player_step + 1) % 2)
        self.update(leaf_value)

    def update(self, value):
        '''
        嵌套在递归函数内，更新当前节点的值
        '''
        self.n += 1
        self.w += value
        self.q = self.w / self.n

    def get_Q(self, c_puct):
        '''
        获取用于选动作的值，UCB公式，一般用于父节点根据子节点的值选择动作，所以不需要判断parent是否为None
        '''
        if self.n == 0:
            u = float("inf")
        else:
            u = c_puct * self.p * np.sqrt(self.parent.n / (1 + self.n))
        return self.q + u

    def is_leaf(self):
        return self.children == {}

    def is_root(self):
        return self.parent == None


class MCTS(object):

    def __init__(self, p_v_fn, c_puct=5, n_playout=1600):
        '''
        p_v_fn: 产生当前节点价值，预估孩子节点的概率
        c_puct: 控制探索
        n_playout: 在每一步进行多少次playout
        '''
        self.root = Node(None, 1.0)
        self.p_v_fn = p_v_fn
        self.c_puct = c_puct
        self.n_playout = n_playout
        

    def playout(self, game):
        node = self.root
        while True:
            if node.is_leaf():
                break
            action, node = node.select(self.c_puct)
            game.step(action % game.dim, action // game.dim)
        available_actions_prob, value = self.p_v_fn(game)
        end, winner = game.is_over()
        player, player_step = game.get_current_player_info()
        if not end:
            node.expand(available_actions_prob)
        else:
            if winner == -1:
                value = 0
            elif (player_step == 1 and player == winner) or (player_step == 0 and player != winner):
                value = 1
            else:
                value = -1
        node.update_recursive(value, player_step)

    def node_move(self, action=-1):
        if action in self.root.children:
            self.root = self.root.children[action]
            self.root.parent = None
        else:
            self.root = Node(None, 1.0)

    def get_action(self, game):
        for i in range(self.n_playout):
            game_copy = copy.deepcopy(game)
            self.playout(game_copy)
        return max(self.root.children.items(), key=lambda item: item[1].n)[0]


class MCTSRL(object):

    def __init__(self, pv_net, c_puct=5, n_playout=1600):
        self.name = pv_net.name
        self.pv_net = pv_net
        self.mcts = MCTS(pv_net.get_probs_and_v, c_puct, n_playout)

    def choose_action(self, game):
        if len(game.available_actions) > 0:
            action = self.mcts.get_action(game)
            self.mcts.node_move(action)
            return action % game.dim, action // game.dim
        else:
            print('棋盘已经满了，无法落子')
