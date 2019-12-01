import numpy as np
import random


def reverse_of(dir_func):
    '''
    反转搜索方向
    '''
    dx, dy = dir_func(0, 0)  # differentiate
    return lambda x, y: (x - dx, y - dy)


class PartialC6(object):
    def __init__(self, env, index=0, threat=4):
        self.env = env
        self.dim = env.box_size               # 棋盘格维度
        self.threat = threat
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
        self.next_owner = [False] * 8
        self.oppo_count = {i: 1 for i in range(8)}
        self.skip_count = [0] * 8
        self.get_8actions = self.get_8actions_v2
        self.next_action = None
        self.jumps = [False] * 8

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
                nx, ny = dir_func(nx, ny)
                if self.is_outta_range(nx, ny):
                    self.next_owner[index] = True
                    index += 1
                    continue
                if self.board[ny][nx] != flag and self.board[ny][nx] != 2:
                    self.next_owner[index] = True
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
            jump2 = False
            if self.actions[index] is not None:
                x, y = self.actions[index]  # 可落子的位置
                nx, ny = func[-1](x, y)  # 下一个位置
                if self.is_outta_range(nx, ny):  # 越界了就退出
                    continue
                if self.board[ny][nx] == 2:  # 如果空位旁边还是空位
                    nx, ny = func[-1](nx, ny)   # 就再往后挪一位
                    jump2 = True    # 标记已经跳了两格了
                    if self.is_outta_range(nx, ny):  # 如果跳2格后越界，就退出
                        continue

                while self.board[ny][nx] == flag:   # 跳1格或者2格之后，如果还是对手的棋，就沿这个方向继续走
                    self.skip_count[index] += 1
                    nx, ny = func[-1](nx, ny)
                    if self.is_outta_range(nx, ny):
                        break

                self.jumps[index] = True

                if self.is_outta_range(nx, ny):  # 如果越界了，就退出
                    continue
                elif self.board[ny][nx] == 2 and jump2 == False:    # 如果遇到空位，空位旁边是对手的棋，并且之前只跳了一格
                    nx, ny = func[-1](nx, ny)   # 就再往后挪一位
                    jump2 = True    # 标记已经跳够2格
                    if self.is_outta_range(nx, ny):  # 如果越界了，就退出
                        continue
                    while self.board[ny][nx] == flag:   # 跳2格后，如果还遇到对手的棋，就沿这个方向继续走
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

    def shuffle_same(self, x: list, y: list):
        '''
        根据列表y中相同元素的位置，排序不变地shuffle列表x
        '''
        a = -1
        b = []
        for i, j in enumerate(y):
            if j != a:
                a = j
                b.append(i)
        b.append(len(y))
        d = []
        for i in range(len(b) - 1):
            c = x[b[i]:b[i + 1]]
            random.shuffle(c)
            d.extend(c)
        return d

    def act(self):
        self.get_8actions()

        def func():
            nums = dict(reversed(sorted(self.oppo_count.items(), key=lambda x: x[1])))
            self.action_index = list(nums.keys())
            self.oppo_nums = list(nums.values())    # 对手在某方向上的落子数
            self.low_threat = True if self.oppo_nums[0] < self.threat else False
            self.action_index = self.shuffle_same(self.action_index, self.oppo_nums)

            list4, list3 = [], []
            for index, value in enumerate(self.oppo_nums):
                if value >= 3:
                    _a = self.actions[self.action_index[index]]
                    if _a is not None:
                        _b = _a[0] + _a[1] * self.dim
                        idx = self.available_actions[_b]
                        if value >= 4:
                            list4.append([int(idx % self.env.dim), int(idx // self.env.dim)])
                        else:
                            list3.append([int(idx % self.env.dim), int(idx // self.env.dim)])

            ergency_idx = []
            for index, value in enumerate(self.oppo_nums):
                if value >= 4:
                    a_idx = self.action_index[index]
                    _a = self.actions[a_idx]
                    if _a is not None:
                        _b = _a[0] + _a[1] * self.dim
                        if self.actions[7 - a_idx] is not None and not self.jumps[a_idx]:
                            if self.next_owner[7 - a_idx] == True:
                                self.actions[a_idx] = None
                                self.actions[7 - a_idx] = None
                                return self.available_actions[_b], False, list4, list3
                            if self.next_owner[a_idx] == True:
                                _c = self.actions[7 - a_idx]
                                _b = _c[0] + _c[1] * self.dim
                                self.actions[a_idx] = None
                                self.actions[7 - a_idx] = None
                                return self.available_actions[_b], False, list4, list3
                            self.next_action = self.actions[7 - a_idx]
                            self.actions[a_idx] = None
                            self.actions[7 - a_idx] = None
                            return self.available_actions[_b], True, list4, list3  # 对方连着四个子， 形式危机

                        if self.next_owner[a_idx] == False:
                            ergency_idx.append(a_idx)

            if len(ergency_idx) >= 2:
                _a = self.actions[ergency_idx[0]]
                _b = _a[0] + _a[1] * self.dim
                self.next_action = self.actions[ergency_idx[1]]
                self.actions[ergency_idx[0]] = None
                self.actions[ergency_idx[1]] = None
                return self.available_actions[_b], True, list4, list3  # 对方有两个4子的，形式危机
            elif len(ergency_idx) == 1:
                _a = self.actions[ergency_idx[0]]
                _b = _a[0] + _a[1] * self.dim
                self.actions[ergency_idx[0]] = None
                return self.available_actions[_b], False, list4, list3    # 对方只有一个4子的，而且一个方向已经被围堵或者四子不相连

            for index, value in enumerate(self.oppo_nums):
                a_idx = self.action_index[index]
                _a = self.actions[a_idx]
                self.actions[a_idx] = None
                if _a is not None:
                    _b = _a[0] + _a[1] * self.dim
                    return self.available_actions[_b], False, list4, list3
            return random.sample(self.env.available_actions, 1)[0], False, list4, list3
        idx, ergency, list4, list3 = func()
        return int(idx % self.env.dim), int(idx // self.env.dim), ergency, list4, list3

    def get_next(self):
        def func():
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
        idx = func()
        return int(idx % self.env.dim), int(idx // self.env.dim)

    def get_low_threat(self):
        return self.low_threat
