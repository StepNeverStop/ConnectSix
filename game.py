import os
import sys
import numpy as np

# console helper methods
def cls():
    os.system('cls' if os.name == 'nt' else 'clear')

def darktext(str):
    return str if os.name == 'nt' else '\x1b[0;30m{}\x1b[0m'.format(str)

def reverse_of(dir_func):
    dx, dy = dir_func(0, 0)  # differentiate
    return lambda x, y: (x-dx, y-dy)

class Connect6(object):

    def __init__(self, dim):
        self.dim = dim
        self.row_info = '  '.join([f'{i+1:>2d}' for i in range(dim)])
        self.directions = {
            'right':        lambda x, y: (x+1, y),
            'right up':     lambda x, y: (x+1, y+1),
            'right down':   lambda x, y: (x+1, y-1),
            'left':         lambda x, y: (x-1, y),
            'left up':      lambda x, y: (x-1, y+1),
            'left down':    lambda x, y: (x-1, y-1),
            'down':         lambda x, y: (x, y-1),
            'up':           lambda x, y: (x, y+1),
        }
    
    def register_players(self, name1, name2):
        """
        name1: 黑子， name2：白子
        """
        assert name1 != name2, "不能给两位选手取相同的名字"
        self.players_name = [name1, name2]
        self.players_char = ['●', '○']
        self.last_char = ['■', '□']

    def render(self):
        """Render交互界面"""
        cls()
        print(' '*50+'珍珑棋局'+' '*50)
        print(f'已经进行了{self.round}回合，{self.total_move}步，黑子{self.players_name[0]} {self.moves[0]}步，白子{self.players_name[1]} {self.moves[1]}步')
        print(f'请 {self.players_name[self.now_player]} 落子！你还有{2-self.move_step}个子可以下。')
        print()
        print('     ' , self.row_info)
        
        for y in range(self.dim):
            print('     ' + '-' * 4 * self.dim)
            print('  {:>2d} |'.format(y+1), end='')  # 行号
            for x in range(self.dim):
                stone = self.board[y][x]
                if stone != 2: 
                    if x == self.last_move[0] and y == self.last_move[1]:
                        print(' ' + self.last_char[self.board[y][x]] + ' ', end='')
                    else:
                        print(' ' + self.players_char[self.board[y][x]] + ' ', end='')
                else: 
                    print(darktext('   '), end='')
                print('|', end='')
            print()

        print('     ' + '-' * 4 * self.dim)

    def reset(self):
        """
        重置环境，返回状态obs
        """
        self.last_move = np.zeros(2, dtype=np.int32)
        self.total_move = 0
        self.round = 1
        self.move_step = 1
        self.board = np.array(
            [np.array(
                [2 for x in range(self.dim)]
                ) for y in range(self.dim)]
            )
        self.now_player = 0
        self.moves = [0, 0]
        return np.eye(3)[self.board].reshape(-1)

    def move(self, x, y):
        """
        返回下一个状态，奖励，是否done
        """
        self.board[y][x] = self.now_player
        self.last_move[0], self.last_move[1] = x, y
        self.total_move += 1
        self.moves[self.now_player] += 1
        self.move_step+=1
        if self.move_step == 2:
            self.round+=1
            self.move_step=0
            self.now_player = (self.now_player+1) % 2
        s = self.get_state()
        return s

    def get_state(self):
        """
        用于在执行一个step步后返回新的状态
        """
        return np.eye(3)[self.board].reshape(-1)

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

    def can_place(self, x, y):
        if self.is_outta_range(x, y):
            return False, '位置越界'
        elif self.board[y][x] != 2:
            return False, f'此处已着子({x+1},{y+1})'
        else:
            return True, '可以落子'

    def is_outta_range(self, x, y):
        return x < 0 or x >= self.dim or y < 0 or y >= self.dim

    def is_over(self):
        board = self.board
        x, y = self.last_move
        for _dir, dir_func in self.directions.items():
            nx, ny = dir_func(x, y)
            if self.is_outta_range(nx, ny): continue

            if board[ny][nx] == board[y][x]:
                # to check properly, go to the end of direction
                while board[ny][nx] == board[y][x]:
                    nx, ny = dir_func(nx, ny)
                    if self.is_outta_range(nx, ny): break

                reverse_dir_func = reverse_of(dir_func)
                nx, ny = reverse_dir_func(nx, ny)  # one step back.

                is_end = self._track(nx, ny, reverse_dir_func)
                if is_end:
                    # returns player who won.
                    return board[ny][nx]


    def _track(self, start_x, start_y, dir_func):
        x, y = start_x, start_y
        original_player = self.board[y][x]

        step = 1
        while True:
            x, y = dir_func(x, y)
            if self.is_outta_range(x, y) or self.board[y][x] != original_player:
                if step == 6: return True
                return False
            step += 1

        if step > 6:
            return False

        return True