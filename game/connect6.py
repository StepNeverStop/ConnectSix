import numpy as np
from .game_base import Game


def reverse_of(dir_func):
    '''
    反转搜索方向
    '''
    dx, dy = dir_func(0, 0)  # differentiate
    return lambda x, y: (x - dx, y - dy)


class Connect6(Game):
    '''
    连六棋游戏
    规则：
        黑棋先手落一子，白棋后手两子，之后每位选手交替两子
        首先在横、竖、斜方向存在相连六个及以上同色棋子的选手获胜
    '''

    def __init__(self, dim):
        super().__init__(dim)
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

    def register(self, player_name1='Player1', player_name2='Player2'):
        super().register(player_name1, player_name2)

    def render(self):
        """Render交互界面"""
        super().render()
        print()
        print(f'已经进行了{self.round}回合，{self.total_move}步，黑子{self.players_name[0]} {self.moves[0]}步，白子{self.players_name[1]} {self.moves[1]}步')
        print(f'请 {self.players_name[self.now_player]} 落子！你还有{2-self.move_step}个子可以下。')

    def reset(self):
        """
        重置环境，返回状态obs
        """
        self.last_move = np.zeros(2, dtype=np.int32)
        self.total_move = 0
        self.round = 1
        self.move_step = 1
        self.board = np.full([self.dim, self.dim], 2)
        self.now_player = 0
        self.moves = [0, 0]
        pass    # 此处可以根据个人需求重写返回信息

    def step(self, x, y):
        """
        执行动作并返回新的棋盘信息
        """
        self.board[y][x] = self.now_player
        self.last_move[0], self.last_move[1] = x, y
        self.total_move += 1
        self.moves[self.now_player] += 1
        self.move_step += 1
        if self.move_step == 2:
            self.round += 1
            self.move_step = 0
            self.now_player = (self.now_player + 1) % 2
        pass    # 此处可以根据个人需求重写返回信息
    
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
        x, y = self.last_move
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
                    # returns player who won.
                    return board[ny][nx]
        if 2 not in self.board:
            return -1

    def _track(self, start_x, start_y, dir_func):
        x, y = start_x, start_y
        original_player = self.board[y][x]

        step = 1
        while True:
            x, y = dir_func(x, y)
            if self.is_outta_range(x, y) or self.board[y][x] != original_player:
                if step == 6:
                    return True
                return False
            step += 1

        if step > 6:
            return False

        return True
