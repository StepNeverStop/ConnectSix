import numpy as np
import random


def reverse_of(dir_func):
    '''
    反转搜索方向
    '''
    dx, dy = dir_func(0, 0)  # differentiate
    return lambda x, y: (x - dx, y - dy)


class PartialC6(object):
    def __init__(self, env, index=0):
        self.env = env
        self.dim = env.box_size               # 棋盘格维度
        self.directions = {
            'up': lambda x, y: (x, y + 1),
            'right': lambda x, y: (x + 1, y),
            'right up': lambda x, y: (x + 1, y + 1),
            'right down': lambda x, y: (x + 1, y - 1),
            'left up': lambda x, y: (x - 1, y + 1),
            'left down': lambda x, y: (x - 1, y - 1),
            'left': lambda x, y: (x - 1, y),
            'down': lambda x, y: (x, y - 1),
        }
        self.board, self.x, self.y, self.available_actions = env.get_partial_board(index=index)
        self.current_player = env.current_player
        self.oppo_player = (env.current_player + 1) % 2
        self.actions = [None] * 8
        self.oppo_count = {i: 1 for i in range(8)}
        self.skip_count = [0] * 8
        self.get_8actions = self.get_8actions_v2
        self.next_action = None

    def is_outta_range(self, x, y):
        return x < 0 or x >= self.dim or y < 0 or y >= self.dim

    def get_8actions_v2(self):
        flag = self.board[self.y][self.x]
        if flag == 2:
            self.actions[0] = [self.x, self.y]
        index = 0
        for _dir, dir_func in self.directions.items():
            nx, ny = dir_func(self.x, self.y)
            if self.is_outta_range(nx, ny):  # 判断x y是不是边界点
                index += 1
                continue

            while self.board[ny][nx] == flag:
                self.oppo_count[index] += 1
                nx, ny = dir_func(nx, ny)
                if self.is_outta_range(nx, ny):
                    break

            if self.is_outta_range(nx, ny):
                index += 1
                continue
            else:
                if self.board[ny][nx] == 2:
                    self.actions[index] = [nx, ny]
            index += 1
        val_list = self.reverse_add(list(self.oppo_count.values()))
        self.oppo_count = {i: val_list[i] for i in range(8)}
        self.calculate_skip(flag)
        val_list = self.add_list(list(self.oppo_count.values()), self.skip_count)
        self.oppo_count = {i: val_list[i] for i in range(8)}

    def add_list(self, x: list, y: list):
        assert len(x) == len(y)
        return (np.array(x) + np.array(y)).tolist()

    def calculate_skip(self, flag):
        for index, func in enumerate(self.directions.items()):
            if self.actions[index] is not None:
                x, y = self.actions[index]
                nx, ny = func[-1](x, y)
                if self.is_outta_range(nx, ny):
                    continue
                if self.board[ny][nx] == 2: # 跳第二步
                    nx, ny = func[-1](nx, ny)
                    if self.is_outta_range(nx, ny):
                        continue
                while self.board[ny][nx] == flag:
                    self.skip_count[index] += 1
                    nx, ny = func[-1](nx, ny)
                    if self.is_outta_range(nx, ny):
                        break

    def reverse_add(self, x: list):
        return (np.array(x) + np.array(list(reversed(x))) - 1).tolist()

    def get_8actions_v1(self):
        board = self.board
        x, y = self.x, self.y
        index = 0
        for _dir, dir_func in self.directions.items():
            nx, ny = dir_func(x, y)
            if self.is_outta_range(nx, ny):  # 判断x y是不是边界点
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

    def shuffle_same(self, x:list, y:list):
        '''
        根据列表y中相同元素的位置，排序不变地shuffle列表x
        '''
        a = -1
        b=[]
        for i, j in enumerate(y):
            if j != a:
                a = j
                b.append(i)
        b.append(len(y))
        d=[]
        for i in range(len(b)-1):
            c = x[b[i]:b[i+1]]
            random.shuffle(c)
            d.extend(c)
        return d

    def act(self):
        self.get_8actions()
        nums = dict(reversed(sorted(self.oppo_count.items(), key=lambda x: x[1])))
        self.action_index = list(nums.keys())
        self.oppo_nums = list(nums.values())
        self.action_index = self.shuffle_same(self.action_index, self.oppo_nums)
        for index, value in enumerate(self.oppo_nums):
            a_idx = self.action_index[index]
            _a = self.actions[a_idx]
            self.actions[a_idx] = None
            if _a is not None:
                _b = _a[0] + _a[1] * self.dim
                if value >= 4:
                    if self.actions[7 - a_idx] is None:
                        return self.available_actions[_b], False
                    else:
                        self.next_action = self.actions[7 - a_idx]
                        self.actions[7 - a_idx] = None
                        return self.available_actions[_b], True
                else:
                    return self.available_actions[_b], False
        return random.sample(self.env.available_actions, 1)[0], False

    def get_next(self):
        if self.next_action is not None:
            x, y = self.next_action
            return self.available_actions[x + y * self.dim]
        for index, value in enumerate(self.oppo_nums):
            a_idx = self.action_index[index]
            _a = self.actions[a_idx]
            self.actions[a_idx] = None
            if _a is not None:
                _b = _a[0] + _a[1] * self.dim
                return self.available_actions[_b]
        return random.sample(self.env.available_actions, 1)[0]

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
