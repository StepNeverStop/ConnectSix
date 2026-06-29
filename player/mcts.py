
import copy

import numpy as np

from .mcts_core import Node


def p_v_fn(game):
    action_probs = np.ones(len(game.available_actions)) / len(game.available_actions)
    return zip(game.available_actions, action_probs), 0


class MCTS:

    def __init__(self, p_v_fn, c_puct=5, n_playout=1600, max_step=1000):
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
        player, player_step = game.get_current_player_info()
        if not end:
            node.expand(available_actions_prob)
            winner = self.rollout2end(game)
        if winner == -1 or winner is None:
            value = 0
        elif (player_step == 1 and player == winner) or (player_step == 0 and player != winner):
            value = 1
        else:
            value = -1
        node.update_recursive(value, player_step)

    def rollout2end(self, game):
        for _ in range(self.max_step):
            end, winner = game.is_over()
            if end:
                break
            action = np.random.choice(game.available_actions)
            x, y = action % game.dim, action // game.dim
            game.step(x, y)
        else:
            print('最大步长，算平局')
        return winner

    def node_move(self, action=-1):
        if action in self.root.children:
            self.root = self.root.children[action]
            self.root.parent = None
        else:
            self.root = Node(None, 1.0)

    def get_action(self, game):
        for _ in range(self.n_playout):
            game_copy = copy.deepcopy(game)
            self.playout(game_copy)
        return max(self.root.children.items(), key=lambda item: item[1].n)[0]


class MCTSPlayer:

    def __init__(self, name, c_puct=5, n_playout=1600, max_step=1000):
        self.name = name
        self.mcts = MCTS(p_v_fn, c_puct, n_playout, max_step)

    def choose_action(self, game):
        if len(game.available_actions) > 0:
            action = self.mcts.get_action(game)
            self.mcts.node_move()
            return action % game.dim, action // game.dim
        print('棋盘已经满了，无法落子')
