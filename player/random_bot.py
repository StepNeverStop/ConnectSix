import random
from .bot_base import Bot

class RandomBot(Bot):
    def __init__(self, dim, name='random_bot'):
        super().__init__(dim, name)

    """ Example bot that runs randomly. """
    def choose_action(self, state):
        x = random.randrange(0, self.dim)
        y = random.randrange(0, self.dim)
        return x, y