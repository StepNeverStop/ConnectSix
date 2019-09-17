
import numpy as np
import copy

def p_v_fn(game):
    action_probs = np.ones(len(game.available_actions))/len(game.available_actions)
    return zip(game.available_actions, action_probs), 0


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
        return max(self.children.items(), key = lambda item: item[-1].get_Q(c_puct))

    def update_recursive(self, leaf_value, player_step):
        '''
        递归函数，向上更新 n, w, q
        '''
        if self.parent:
            self.parent.update_recursive(leaf_value if player_step==0 else -leaf_value, (player_step+1)%2)
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
        u = c_puct * self.p * np.sqrt(self.parent.n / (1 + self.n))
        return self.q + u

    def is_leaf(self):
        return self.children == {}

    def is_root(self):
        return self.parent == None

class MCTS(object):

    def __init__(self, p_v_fn, c_puct=5, n_playout=1600, max_step=1000):
        '''
        p_v_fn: 产生当前节点价值，预估孩子节点的概率
        c_puct: 控制探索
        n_playout: 在每一步进行多少次playout
        max_step: 在每个playout向前预估多少步
        '''
        self.root = Node(None, 1.0)
        self.p_v_fn = p_v_fn
        self.c_puct = c_puct
        self.n_playout = n_playout
        self.max_step = max_step
    
    def playout(self, game):
        node = self.root
        while True:
            if node.is_leaf():
                break
            action, node = node.select(self.c_puct)
            game.step(action % game.dim, action // game.dim)
        available_actions_prob, _ = self.p_v_fn(game)
        end, winner = game.is_over()
        if not end:
            node.expand(available_actions_prob)
        playout_value, player_step = self.rollout2end(game)
        node.update_recursive(playout_value, player_step)

    
    def rollout2end(self, game):
        player, player_step = game.get_current_player_info()
        for i in range(self.max_step):
            end, winner = game.is_over()
            if end:
                break
            action = np.random.choice(game.available_actions)
            x, y = action % game.dim, action // game.dim
            game.step(x, y)
        else:
            print('最大步长，算平局')
        if winner==-1 or winner==None:
            value = 0
        else:
            if (player_step==1 and player==winner) or (player_step ==0 and player!=winner):
                value = 1
            else:
                value = -1
        return value, player_step

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


class MCTSPlayer(object):

    def __init__(self, name, c_puct=5, n_playout=1600, max_step=1000):
        self.name = name
        self.mcts = MCTS(p_v_fn, c_puct, n_playout, max_step)

    def choose_action(self, game):
        if len(game.available_actions) > 0:
            action = self.mcts.get_action(game)
            self.mcts.node_move()
            return action % game.dim, action // game.dim
        else:
            print('棋盘已经满了，无法落子')