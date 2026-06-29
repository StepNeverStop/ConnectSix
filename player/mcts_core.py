"""Shared MCTS tree node and utility functions."""

import numpy as np


def softmax(x):
    probs = np.exp(x - np.max(x))
    probs /= np.sum(probs)
    return probs


class Node:
    """MCTS tree node with UCB selection."""

    def __init__(self, parent, prior_prob):
        self.parent = parent
        self.children = {}
        self.n = 0
        self.q = 0
        self.w = 0
        self.p = prior_prob

    def expand(self, available_actions_prob):
        for action, prob in available_actions_prob:
            if action not in self.children:
                self.children[action] = Node(self, prob)

    def select(self, c_puct):
        return max(self.children.items(), key=lambda item: item[-1].get_Q(c_puct))

    def update_recursive(self, leaf_value, player_step):
        if self.parent:
            self.parent.update_recursive(
                leaf_value if player_step == 0 else -leaf_value,
                (player_step + 1) % 2,
            )
        self.update(leaf_value)

    def update(self, value):
        self.n += 1
        self.w += value
        self.q = self.w / self.n

    def get_Q(self, c_puct):
        if self.n == 0:
            u = float("inf")
        else:
            u = c_puct * self.p * np.sqrt(self.parent.n / (1 + self.n))
        return self.q + u

    def is_leaf(self):
        return self.children == {}

    def is_root(self):
        return self.parent is None
