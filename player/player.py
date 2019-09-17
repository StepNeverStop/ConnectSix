from .bot_base import Bot

class Player(Bot):
    """
    人类控制的棋手
    """
    def __init__(self, dim, name='human_player'):
        super().__init__(dim, name)

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