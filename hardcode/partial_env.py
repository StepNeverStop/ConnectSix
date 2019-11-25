import numpy as np


def reverse_of(dir_func):
    '''
    反转搜索方向
    '''
    dx, dy = dir_func(0, 0)  # differentiate
    return lambda x, y: (x - dx, y - dy)


class PartialC6(object):
    def __init__(self, env, dim=11, index=0):
        self.dim = dim               # 棋盘格维度
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
        self.board, self.x, self.y, self.available_actions = env.get_partial_board(box_size=self.dim, index=index)
        self.current_player = env.current_player
        self.oppo_player = (env.current_player + 1) % 2
        self.actions = [None] * 8
        self.oppo_count = {i: 1 for i in range(8)}

    def is_outta_range(self, x, y):
        return x < 0 or x >= self.dim or y < 0 or y >= self.dim

    def get_8actions(self):
        board = self.board
        x, y = self.x, self.y
        index = 0
        for _dir, dir_func in self.directions.items():
            nx, ny = dir_func(x, y)
            if self.is_outta_range(nx, ny):
                index += 1
                continue

            while board[ny][nx] == board[y][x]:
                self.oppo_count[index] += 1
                nx, ny = dir_func(nx, ny)
                if self.is_outta_range(nx, ny):
                    break

            if self.is_outta_range(nx, ny):
                index += 1
                continue
            else:
                if board[ny][nx] == 2:
                    self.actions[index] = [nx, ny]
                    nx, ny = dir_func(nx, ny)
                    if self.is_outta_range(nx, ny):
                        index += 1
                        continue
                    while board[ny][nx] == board[y][x]:
                        self.oppo_count[index] += 1
                        nx, ny = dir_func(nx, ny)
                        if self.is_outta_range(nx, ny):
                            break
            index += 1

    def act(self):
        self.get_8actions()
        nums = dict(reversed(sorted(self.oppo_count.items(), key=lambda x: x[1])))
        for key, value in nums.items():
            if self.actions[key] is not None:
                return self.actions[key]

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
