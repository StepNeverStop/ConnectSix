import os
import numpy as np
from abc import ABC, abstractmethod

def cls():  # console helper methods
    os.system('cls' if os.name == 'nt' else 'clear')


def darktext(str):
    return str if os.name == 'nt' else '\x1b[0;30m{}\x1b[0m'.format(str)


class Game(ABC):
    """
    棋盘游戏基类，棋盘必须为方阵，所以设置一个维度就可以了
    双人对战棋类
    黑棋先行，白棋后手，黑棋用下标0表示，白棋用下标1或-1表示
    """
    def __init__(self, dim):
        self.dim = dim                                      # 棋盘格维度
        self.players_char = ['●', '○']                      # 用于显示黑棋、白棋的样式
        self.player_order = ['黑棋', '白棋']                 # 定义下棋顺序
        self.board = np.full([self.dim, self.dim], -1)      # 用于初始化棋盘
        self.last_move = np.zeros(2, dtype=np.int32)        # 用于记录最新的落子信息
        self.last_move_char = ['■', '□']                    # 用于显示黑棋、白棋刚刚落子的样式，为了引起注意
        self.total_move = 0                                 # 用于统计该盘棋过走了多少步
        self.round = 1                                      # 用于统计目前进行到了第几回合，因为有些棋类会出现一回合进行多步操作的情况，所以可能会存在total_move与round不等的情况
        self.now_player = 0                                 # 用于记录目前落子的选手是黑子还是白子
        self.moves = [0, 0]                                 # 用于分别统计两位选手的步数
        self.row_info = '  '.join([f'{i+1:>2d}' for i in range(dim)])   # 用于渲染时显示棋盘上部数字上标
        self.register()
    
    def register(self, player_name1='Player1', player_name2='Player2'):
        """
        用于注册两位选手的名字，因为在render函数会需要选手名字的信息，所以可以在游戏开始前通过该函数将选手信息传递进来
        player_name1: 黑子， player_name2: 白子
        """
        assert player_name1 != player_name2, "不能给两位选手取相同的名字"
        self.players_name = [player_name1, player_name2]

    def render(self) -> None:
        '''
        用于编写渲染显示的函数
        '''
        """Render交互界面"""
        cls()
        print(' ' * 30 + '珍珑棋局' + ' ' * 30) # 标题
        print()
        print('     ', self.row_info)   # 棋盘顶部数字标号

        for y in range(self.dim):
            print('     ' + '-' * 4 * self.dim) # 横线界
            print('  {:>2d} |'.format(y + 1), end='')  # 行号
            for x in range(self.dim):   # 显示棋子样式
                stone = self.board[y][x]
                if stone != 2:
                    if x == self.last_move[0] and y == self.last_move[1]:
                        print(' ' + self.last_move_char[self.board[y][x]] + ' ', end='')
                    else:
                        print(' ' + self.players_char[self.board[y][x]] + ' ', end='')
                else:
                    print(darktext('   '), end='')
                print('|', end='')  #竖线界
            print()

        print('     ' + '-' * 4 * self.dim) # 最底部横线界
    
    @abstractmethod
    def reset(self):
        '''
        用于在每盘棋开始之前的初始化操作
        '''
        pass

    @abstractmethod
    def step(self, x, y):
        '''
        一般将游戏控制逻辑，rule写在此处，更新用于统计信息的变量并转换棋手信息，对于逻辑复杂的棋盘规则需要在类外部实现与该函数内相应的逻辑，已达到规则同步
        用于对某些学习算法执行动作后返回相应的反馈信息，如棋盘最新状态、立即奖励、游戏结束信息等
        '''
        pass

    @abstractmethod
    def is_over(self):
        '''
        用于判断棋盘是否已经出现赢家，或者和局
        因为一般棋类只有对手下子致使我方输棋的情况，即不存在我方走一子然后判输的情况，故可直接根据当前选手的信息判定孰胜孰负，不需额外传输player信息
        '''
        pass
