import numpy as np
from itertools import combinations, product
import multiprocessing
import os
import time
import math


ratio = [
    [1, 10, 15, 35, 25, 10000],
    [1, 10, 15, 350, 250, 10000],
    [1, 10, 15, 35, 25, 10000]
]
opponent_ratio = [
    [1, 10, 15, 35, 25, 10000],
    [1, 10, 15, 35, 25, 10000],
    [1, 10, 15, 350, 250, 10000]
]
directions = {
    'up': lambda x, y: (x, y + 1),
    'right': lambda x, y: (x + 1, y),
    'right up': lambda x, y: (x + 1, y + 1),
    'right down': lambda x, y: (x + 1, y - 1),
    'left': lambda x, y: (x - 1, y),
    'left up': lambda x, y: (x - 1, y + 1),
    'left down': lambda x, y: (x - 1, y - 1),
    'down': lambda x, y: (x, y - 1),
}


def is_outta_range(x, y, dim):
    return x < 0 or x >= dim or y < 0 or y >= dim


def is_in_pos_range(x, y, pos, bound):
    return x > pos[0] - bound and y > pos[1] - bound and x < pos[0] + bound and y < pos[1] + bound


def get_available_pos(board):
    reserverd_available_pos = []
    existed_pos = zip(*np.where(board != 2))  # 已经落子的点
    available_single_pos = []  # 可以落子的点（在主战场附近） [[x1,y1],[x2,y2],...]
    for pos in existed_pos:
        for dir_func in directions.values():
            nx, ny = dir_func(*pos)

            if not is_outta_range(nx, ny, board.shape[0]) and board[nx, ny] == 2:
                if [nx, ny] not in available_single_pos:
                    available_single_pos.append([nx, ny])

                nnx, nny = dir_func(nx, ny)

                if not is_outta_range(nnx, nny, board.shape[0]) and board[nnx, nny] == 2 and [nnx, nny] not in available_single_pos:
                    reserverd_available_pos.append(([nx, ny], [nnx, nny]))

    available_pos = list(combinations(available_single_pos, 2))  # 可以落子的两两组合 [([x1,y1],[x2,y2]), ([x1,y1],[x3,y3]), ...]
    return available_pos + reserverd_available_pos, available_single_pos


def get_available_single_pos_around(board, pos):
    reserverd_available_pos = []
    existed_pos = zip(*np.where(board != 2))  # 已经落子的点
    available_single_pos = []  # 可以落子的点（在主战场附近） [[x1,y1],[x2,y2],...]

    for dir_func in directions.values():
        nx, ny = dir_func(*pos)

        if not is_outta_range(nx, ny, board.shape[0]) and board[nx, ny] == 2:
            if [nx, ny] not in available_single_pos:
                available_single_pos.append([nx, ny])

            nnx, nny = dir_func(nx, ny)

            if not is_outta_range(nnx, nny, board.shape[0]) and board[nnx, nny] == 2 and [nnx, nny] not in available_single_pos:
                reserverd_available_pos.append(([nx, ny], [nnx, nny]))

    # pos附近的单点, 保留点
    return available_single_pos, reserverd_available_pos


def get_sum_value_from_board_values(board_values, opponent_board_values, ratio_i):
    total_value = 0
    opponent_total_value = 0
    for dir_num in range(4):  # 计算每条路的变更值
        for i in range(6):
            c = np.sum(board_values[dir_num] == (i + 1))
            total_value += c * ratio[ratio_i][i]
            c = np.sum(opponent_board_values[dir_num] == (i + 1))
            opponent_total_value += c * opponent_ratio[ratio_i][i]

    return total_value - opponent_total_value


def _process_get_available_pos_values(pipe):
    while True:
        board, available_pos, player, board_values, opponent_board_values, ratio_i = pipe.recv()
        available_pos_values = []

        for pos in available_pos:
            new_board, new_board_values, new_opponent_board_values = update_board_values(board, pos, player, board_values, opponent_board_values)

            value = get_sum_value_from_board_values(new_board_values, new_opponent_board_values, ratio_i)

            available_pos_values.append(int(value))

        pipe.send(available_pos_values)


def get_line(board, pos, dir_num, bound=6):  # 获取dir_num方向的以pos为中心的坐标
    dir_1, dir_2 = [[1, 0], [0, 1], [1, 1], [1, -1]][dir_num]

    line = []

    ni, nj = pos
    line.append((ni, nj))

    ni, nj = pos[0] - dir_1, pos[1] - dir_2
    while not is_outta_range(ni, nj, board.shape[0]) and is_in_pos_range(ni, nj, pos, bound):
        line.insert(0, (ni, nj))
        ni -= dir_1
        nj -= dir_2

    ni, nj = pos[0] + dir_1, pos[1] + dir_2
    while not is_outta_range(ni, nj, board.shape[0]) and is_in_pos_range(ni, nj, pos, bound):
        line.append((ni, nj))
        ni += dir_1
        nj += dir_2

    return line, [board[p[0], p[1]] for p in line]


def update_board_value(line, line_v, player, board_value):
    opponent_player = (player + 1) % 2

    for i, _ in enumerate(line_v[:-5]):
        if line_v[i:i + 6].count(opponent_player) == 0:
            board_value[line[i][0], line[i][1]] = line_v[i:i + 6].count(player)
        else:
            board_value[line[i][0], line[i][1]] = 0


def update_board_values(board, pos, player, board_values, opponent_board_values):
    p1, p2 = pos
    new_board = board.copy()
    new_board[p1[0], p1[1]] = player
    new_board[p2[0], p2[1]] = player
    new_board_values = [v.copy() for v in board_values]
    new_opponent_board_values = [v.copy() for v in opponent_board_values]

    for dir_num in range(4):  # 计算每条路的变更值
        board_value = new_board_values[dir_num]
        opponent_board_value = new_opponent_board_values[dir_num]

        for p in [p1, p2]:
            line, line_v = get_line(new_board, p, dir_num)
            update_board_value(line, line_v, player, board_value)  # 更新player价值表
            update_board_value(line, line_v, (player + 1) % 2, opponent_board_value)  # 更新(player + 1) % 2价值表

    # print(np.sum(new_board_values == 1))
    return new_board, new_board_values, new_opponent_board_values


class ZZY_Bot:
    def __init__(self, client, dim, player):
        self.client = client
        self.dim = dim
        self.player = player

        pipe_tuples = [multiprocessing.Pipe() for i in range(8)]
        self._pipes = [p[0] for p in pipe_tuples]
        for p in pipe_tuples:
            multiprocessing.Process(target=_process_get_available_pos_values,
                                    args=(p[1],)).start()

    def get_available_pos_values_multiprocess(self, board, available_pos, player, board_values, opponent_board_values, ratio_i):
        available_pos_values = [0] * len(available_pos)

        seg = math.ceil(len(available_pos) / len(self._pipes))

        for i, pipe in enumerate(self._pipes):
            pipe.send((board, available_pos[seg * i:seg * (i + 1)],
                       player, board_values, opponent_board_values, ratio_i))

        for i, pipe in enumerate(self._pipes):
            tmp_available_pos_values = pipe.recv()
            for j, v in enumerate(tmp_available_pos_values):
                available_pos_values[j + seg * i] = v

        return available_pos_values

    def get_board_values(self, board):
        player_board_values = [np.zeros([self.dim, self.dim], dtype=np.int32), np.zeros([self.dim, self.dim], dtype=np.int32),
                               np.zeros([self.dim, self.dim], dtype=np.int32), np.zeros([self.dim, self.dim], dtype=np.int32)]
        opponent_player_board_values = [np.zeros([self.dim, self.dim], dtype=np.int32), np.zeros([self.dim, self.dim], dtype=np.int32),
                                        np.zeros([self.dim, self.dim], dtype=np.int32), np.zeros([self.dim, self.dim], dtype=np.int32)]
        for player in [0, 1]:
            if player == self.player:
                board_values = player_board_values
            else:
                board_values = opponent_player_board_values

            for dir_num in range(2):
                board_value = board_values[dir_num]
                for i in range(self.dim):
                    if dir_num == 0:
                        line, line_v = get_line(board, [0, i], dir_num, self.dim)
                    elif dir_num == 1:
                        line, line_v = get_line(board, [i, 0], dir_num, self.dim)

                    update_board_value(line, line_v, player, board_value)

            for dir_num in range(2, 4):
                board_value = board_values[dir_num]
                for i in range(self.dim):
                    line, line_v = get_line(board, [0, i], dir_num, self.dim)

                    update_board_value(line, line_v, player, board_value)

                for i in range(1, self.dim):
                    if dir_num == 2:
                        line, line_v = get_line(board, [i, 0], dir_num, self.dim)
                    elif dir_num == 3:
                        line, line_v = get_line(board, [i, self.dim - 1], dir_num, self.dim)

                    update_board_value(line, line_v, player, board_value)

        return player_board_values, opponent_player_board_values

    def choose_action(self, board, last_move=None):
        if np.all(board == 2):  # 下第一步棋
            x, y = [board.shape[0] // 2 + 1], [board.shape[1] // 2 + 1]
            if self.client is not None:
                self.client.move(x, y)
            return x, y

        board_values, opponent_board_values = self.get_board_values(board)

        base_value = get_sum_value_from_board_values(board_values, opponent_board_values, 0)
        print('base_value', base_value)
        if base_value >= 0:
            ratio_i = 1  # 进攻
        else:
            ratio_i = 2  # 防守

        available_pos, available_single_pos = get_available_pos(board)

        available_pos_values = self.get_available_pos_values_multiprocess(board, available_pos, self.player,
                                                                          board_values, opponent_board_values, ratio_i)

        available_pos_opponent_values = self.get_available_pos_values_multiprocess(board, available_pos, (self.player + 1) % 2,
                                                                                   opponent_board_values, board_values, 1)

        next_opponent_max_values = [0] * len(available_pos)
        approx_available_pos_values = available_pos_values[:]

        # 近似搜索
        for i, check_pos in enumerate(available_pos):
            tmp_next_opponent_values = []
            for j, pos in enumerate(available_pos):
                if pos[0] not in check_pos and pos[1] not in check_pos:  # TODO 扩大范围
                    # if pos[0] == [22,24] and pos[1] == [22, 26]:
                    #     print(available_pos_opponent_values[j])
                    tmp_next_opponent_values.append(available_pos_opponent_values[j])

            next_opponent_max_values[i] = max(tmp_next_opponent_values)
            approx_available_pos_values[i] -= next_opponent_max_values[i]

        # 降序排列
        pos_value_tuple = list(zip(available_pos, available_pos_values, next_opponent_max_values, approx_available_pos_values))
        pos_value_tuple.sort(key=lambda t: t[3], reverse=True)
        available_pos, available_pos_values, next_opponent_max_values, _ = zip(*pos_value_tuple)
        available_pos, available_pos_values, next_opponent_max_values = list(available_pos), list(available_pos_values), list(next_opponent_max_values)
        # for i in range(len(available_pos)):
        #     print(available_pos[i], available_pos_values[i], approx_available_pos_values[i], next_opponent_max_values[i])

        available_pos = available_pos[:50]
        available_pos_values = available_pos_values[:50]
        next_opponent_max_values = next_opponent_max_values[:50]

        for i, pos in enumerate(available_pos):
            new_board, new_board_values, new_opponent_board_values = update_board_values(board, pos, self.player, board_values, opponent_board_values)
            new_available_single_pos, new_reserverd_available_pos = [], []
            for p in pos:
                tmp_new_available_single_pos, tmp_new_reserverd_available_pos = get_available_single_pos_around(new_board, p)
                for pp in tmp_new_available_single_pos:
                    if pp not in new_available_single_pos:
                        new_available_single_pos.append(pp)
                for pp in tmp_new_reserverd_available_pos:
                    if pp not in new_reserverd_available_pos:
                        new_reserverd_available_pos.append(pp)

            old_available_single_pos = [p for p in available_single_pos if p not in pos]  # TODO 扩大范围
            new_available_pos = list(product(new_available_single_pos, old_available_single_pos))
            new_available_pos = new_available_pos + new_reserverd_available_pos
            new_available_pos_values = self.get_available_pos_values_multiprocess(new_board,
                                                                                  new_available_pos,
                                                                                  (self.player + 1) % 2,
                                                                                  new_opponent_board_values,
                                                                                  new_board_values, 1)
            max_new_available_pos_value = max(new_available_pos_values)
            if max_new_available_pos_value > next_opponent_max_values[i]:
                next_opponent_max_values[i] = max_new_available_pos_value

            available_pos_values[i] -= next_opponent_max_values[i]

        idx = available_pos_values.index(max(available_pos_values))
        if idx != 0:
            print('!!!!!', idx)
        pos = available_pos[idx]
        x, y = [pos[0][0], pos[1][0]], [pos[0][1], pos[1][1]]
        if self.client is not None:
            self.client.move(x, y)
        return x, y
