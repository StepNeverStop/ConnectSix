import numpy as np


def reverse_of(dir_func):
    '''
    反转搜索方向
    '''
    dx, dy = dir_func(0, 0)  # differentiate
    return lambda x, y: (x - dx, y - dy)


class Connect6():
    '''
    连六棋游戏
    规则：
        黑棋先手落一子，白棋后手两子，之后每位选手交替两子
        首先在横、竖、斜方向存在相连六个及以上同色棋子的选手获胜
    '''

    def __init__(self, dim):
        self.dim = dim         # 棋盘格维度

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

        self.register()
        self.reset()

    def register(self, player_name1='Player1', player_name2='Player2'):
        assert player_name1 != player_name2, "不能给两位选手取相同的名字"
        self.players_name = [player_name1, player_name2]

    def reset(self):
        self.board = np.full([self.dim, self.dim], 2)  # 用于初始化棋盘

        self.last_move = ([], [])
        self.total_move = 0  # 用于统计该盘棋过走了多少步

        self.current_player = 0
        self.moves = [0, 0]

    def get_current_player(self):
        return self.current_player

    def get_current_state(self):
        return self.board, self.last_move

    def step(self, x, y):
        """
        [x1, x2], [y1, y2]
        执行动作并返回新的棋盘信息
        """
        for i in range(len(x)):
            b, info = self.can_place(x[i], y[i])
            if b:
                self.board[x[i], y[i]] = self.current_player
            else:
                print(info)
        self.total_move += 1
        self.last_move = (x, y)

        self.current_player = (self.current_player + 1) % 2

        return self.last_move

    '''
    以下为游戏规则逻辑，不需要修改
    '''

    def can_place(self, x, y):
        """
        判断该动作是否可以执行
        """
        if self.is_outta_range(x, y):
            return False, '位置越界'
        elif self.board[x][y] != 2:
            return False, f'此处已着子({x},{y})'
        else:
            return True, '可以落子'

    def is_outta_range(self, x, y):
        return x < 0 or x >= self.dim or y < 0 or y >= self.dim

    def is_over(self):
        """
        判断游戏是否已经结束
        """
        board = self.board
        _x, _y = self.last_move
        for i in range(len(_x)):
            x, y = _x[i], _y[i]
            for _dir, dir_func in self.directions.items():
                nx, ny = dir_func(x, y)
                if self.is_outta_range(nx, ny):
                    continue

                if board[nx][ny] == board[x][y]:
                    # to check properly, go to the end of direction
                    while board[nx][ny] == board[x][y]:
                        nx, ny = dir_func(nx, ny)
                        if self.is_outta_range(nx, ny):
                            break

                    reverse_dir_func = reverse_of(dir_func)
                    nx, ny = reverse_dir_func(nx, ny)  # one step back.

                    is_end = self._track(nx, ny, reverse_dir_func)
                    if is_end:
                        if board[nx][ny] == 2:
                            return False, None
                        else:
                            # returns player who won.
                            return True, board[nx][ny]

            if 2 not in self.board:
                return True, -1
            else:
                return False, None

    def _track(self, start_x, start_y, dir_func):
        x, y = start_x, start_y
        original_player = self.board[x][y]

        step = 1
        while True:
            x, y = dir_func(x, y)
            if self.is_outta_range(x, y) or self.board[x][y] != original_player:
                if step >= 6:   # 同色连子数大于等于6个时，判定为胜
                    return True
                return False
            step += 1

        if step > 6:
            return True

        return True
