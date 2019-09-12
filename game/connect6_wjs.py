import numpy as np
from .connect6 import Connect6

class Connect6WJS(Connect6):

    def __init__(self, dim):
        super().__init__(dim)

    def reset(self):
        super().reset()
        return self.board.reshape(-1)

    def step(self, x, y):
        """
        执行动作并返回新的棋盘信息
        """
        super().step(x, y)
        s = self.get_state()
        return s

    def get_state(self):
        """
        用于在执行一个step步后返回新的状态，可以自行定义修改，也可以在AI接收到状态后自行解析成自己希望的输入状态
        """
        return self.board.reshape(-1)

    def get_reward(self):
        """
        用于定义奖励函数
        """
        result = self.is_over()
        if result is not None:
            return 1
        else:
            return 0

    def is_done(self):
        """
        用于判断是否返回done
        """
        result = self.is_over()
        if result is not None:
            return True
        else:
            return False