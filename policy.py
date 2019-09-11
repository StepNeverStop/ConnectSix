from bot import Bot

class MyBot(Bot):
    """
    实现自己的智能体策略
    """
    def __init__(self, dim):
        super().__init__(dim)
        pass

    def store(self, **args):
        pass

    def choose_action(self, state):
        pass

    def learn(self):
        pass