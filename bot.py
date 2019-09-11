
import random

class Bot:
    """ A bot class. Override this class to implement your own Connect6 AI. """

    def __init__(self, dim = 19):
        assert type(dim) == int and dim > 0
        self.dim = dim

    def choose_action(self, state):
        raise NotImplementedError("Implement this to build your own AI.")
        pass
    
    def learn(self):
        pass

    def store(self, **args):
        pass

    def learn(self):
        pass


class RandomBot(Bot):

    """ Example bot that runs randomly. """
    def choose_action(self, state):
        x = random.randrange(0, self.dim)
        y = random.randrange(0, self.dim)
        return x, y

class Player(Bot):

    def choose_action(self, state):
        while True:
            move = input('落子吧: ')
            try:
                xx, yy = move.split('-')
                x, y = (int(xx)-1) % self.dim, (int(yy)-1) % self.dim
            except Exception as e:
                print(e)
                continue
            else:
                return x, y