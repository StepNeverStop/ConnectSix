import os
import numpy as np


def cls():  # console helper methods
    os.system('cls' if os.name == 'nt' else 'clear')


def darktext(str):
    return str if os.name == 'nt' else '\x1b[0;30m{}\x1b[0m'.format(str)


def reverse_of(dir_func):
    '''
    反转搜索方向
    '''
    dx, dy = dir_func(0, 0)  # differentiate
    return lambda x, y: (x - dx, y - dy)


class Connect6(object):
    '''
    黑棋先行，白棋后手，黑棋用下标0表示，白棋用下标1或-1表示

    连六棋游戏
    规则：
        黑棋先手落一子，白棋后手两子，之后每位选手交替两子
        首先在横、竖、斜方向存在相连六个及以上同色棋子的选手获胜
    '''

    def __init__(self, dim=19, box_size=11):
        self.dim = dim              # 棋盘格维度
        self.box_size = box_size
        self.players_char = ['●', '○']                      # 用于显示黑棋、白棋的样式
        self.player_order = ['黑棋', '白棋']                 # 定义下棋顺序
        self.last_move_char = ['■', '□']                    # 用于显示黑棋、白棋刚刚落子的样式，为了引起注意
        self.row_info = '  '.join([f'{i:>2d}' for i in range(self.dim)])   # 用于渲染时显示棋盘上部数字上标
        self.directions = {
            'up': lambda x, y: (x, y + 1),
            'right': lambda x, y: (x + 1, y),
            'right up': lambda x, y: (x + 1, y + 1),
            'right down': lambda x, y: (x + 1, y - 1),
            'left': lambda x, y: (x - 1, y),
            'left up': lambda x, y: (x - 1, y + 1),
            'left down': lambda x, y: (x - 1, y - 1),
            'down': lambda x, y: (x, y - 1),
        }

        self.board = np.full([self.dim, self.dim], 2)       # 用于初始化棋盘
        self.last_move = np.full((2, 2), -1, dtype=np.int32)   # 用于记录最新的落子信息
        self.total_move = 0                                 # 用于统计该盘棋过走了多少步
        self.round = 1                                      # 用于统计目前进行到了第几回合，因为有些棋类会出现一回合进行多步操作的情况，所以可能会存在total_move与round不等的情况
        self.current_player = 0                             # 用于记录目前落子的选手是黑子还是白子
        self.moves = [0, 0]                                 # 用于分别统计两位选手的步数
        self.available_actions = list(range(pow(self.dim, 2)))

    def get_partial_board(self, index=0):
        x, y = self.last_move[index]
        if x == -1 or y == -1:
            x, y = self.last_move[(index+1)%2]
        mid = int((self.box_size - 1) / 2)
        minc = int((self.box_size - 1) / 2)
        maxc = int(self.dim - (self.box_size + 1) / 2)
        if minc <= x <= maxc:
            _x = mid
        elif x < minc:
            _x = x
            x = minc
        elif x > maxc:
            _x = x - self.dim + self.box_size
            x = maxc

        if minc <= y <= maxc:
            _y = mid
        elif y < minc:
            _y = y
            y = minc
        elif y > maxc:
            _y = y - self.dim + self.box_size
            y = maxc
        ltx, lty = x - minc, y - minc
        board = self.board[..., lty:lty + self.box_size, ltx:ltx + self.box_size]
        ad = {}
        for i in range(self.box_size):   # x
            for j in range(self.box_size):   # y
                if self.board[lty + j][ltx + i] == 2:
                    ad[i + j * self.box_size] = (ltx + i) + (lty + j) * self.dim
        return board, _x, _y, ad

    def reset(self):
        """
        重置环境，返回状态obs
        """
        self.last_move = np.full((2, 2), -1, dtype=np.int32)
        self.total_move = 0
        self.round = 1
        self.move_step = 1
        self.board = np.full([self.dim, self.dim], 2)
        self.current_player = 0
        self.moves = [0, 0]
        self.available_actions = list(range(pow(self.dim, 2)))

    def render(self) -> None:
        '''
        用于编写渲染显示的函数
        '''
        """Render交互界面"""
        cls()
        print(' ' * 30 + '珍珑棋局' + ' ' * 30)  # 标题
        print()
        print('     ', self.row_info)   # 棋盘顶部数字标号

        for y in range(self.dim):
            print('     ' + '-' * 4 * self.dim)  # 横线界
            print('  {:>2d} |'.format(y), end='')  # 行号
            for x in range(self.dim):   # 显示棋子样式
                stone = self.board[y][x]
                if stone != 2:
                    if x == self.last_move[(self.move_step + 1) % 2][0] and y == self.last_move[(self.move_step + 1) % 2][1]:
                        print(' ' + self.last_move_char[self.board[y][x]] + ' ', end='')
                    else:
                        print(' ' + self.players_char[self.board[y][x]] + ' ', end='')
                else:
                    print(darktext('   '), end='')
                print('|', end='')  # 竖线界
            print()

        print('     ' + '-' * 4 * self.dim)  # 最底部横线界

    def get_current_player_info(self):
        return self.current_player, self.move_step

    def get_current_state(self):
        return self.board[..., np.newaxis]

    def step(self, x, y):
        """
        执行动作并返回新的棋盘信息
        """
        self.available_actions.remove(x + y * self.dim)
        self.board[y][x] = self.current_player
        self.total_move += 1
        self.last_move[self.move_step] = [x, y]
        self.moves[self.current_player] += 1
        self.move_step += 1
        if self.move_step == 2:
            self.round += 1
            self.move_step = 0
            self.current_player = (self.current_player + 1) % 2

    def step_again(self):
        if self.move_step != 1:
            return
        x, y = self.last_move[0]
        self.total_move += 1
        self.moves[self.current_player] += 1
        self.last_move[self.move_step] = [x, y]
        self.move_step += 1
        if self.move_step == 2:
            self.round += 1
            self.move_step = 0
            self.current_player = (self.current_player + 1) % 2
    '''
    以下为游戏规则逻辑，不需要修改
    '''

    def can_place(self, x, y):
        """
        判断该动作是否可以执行
        """
        if self.is_outta_range(x, y):
            return False, '位置越界'
        elif self.board[y][x] != 2:
            return False, f'此处已着子({x+1},{y+1})'
        else:
            return True, '可以落子'

    def is_outta_range(self, x, y):
        return x < 0 or x >= self.dim or y < 0 or y >= self.dim

    def is_over(self):
        """
        判断游戏是否已经结束
        """
        board = self.board
        x, y = self.last_move[(self.move_step + 1) % 2]
        for _dir, dir_func in self.directions.items():
            nx, ny = dir_func(x, y)
            if self.is_outta_range(nx, ny):
                continue

            if board[ny][nx] == board[y][x]:
                # to check properly, go to the end of direction
                while board[ny][nx] == board[y][x]:
                    nx, ny = dir_func(nx, ny)
                    if self.is_outta_range(nx, ny):
                        break

                reverse_dir_func = reverse_of(dir_func)
                nx, ny = reverse_dir_func(nx, ny)  # one step back.

                is_end = self._track(nx, ny, reverse_dir_func)
                if is_end:
                    if board[ny][nx] == 2:
                        return False, None
                    else:
                        # returns player who won.
                        return True, board[ny][nx]
        if 2 not in self.board:
            return True, -1
        else:
            return False, None

    def _track(self, start_x, start_y, dir_func):
        x, y = start_x, start_y
        original_player = self.board[y][x]

        step = 1
        while True:
            x, y = dir_func(x, y)
            if self.is_outta_range(x, y) or self.board[y][x] != original_player:
                if step >= 6:   # 同色连子数大于等于6个时，判定为胜
                    return True
                return False
            step += 1

        if step > 6:
            return True

        return True
