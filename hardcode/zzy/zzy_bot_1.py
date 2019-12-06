import numpy as np
from itertools import combinations


class ZZY_Bot_1:
    def __init__(self, client, dim, player):
        self.client = client
        self.dim = dim
        self.player = player

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

        self.ratio = [1, 10, 15, 350, 250, 10000]
        self.opponent_ratio = [1, 10, 15, 35, 25, 10000]

    def is_outta_range(self, x, y):
        return x < 0 or x >= self.dim or y < 0 or y >= self.dim

    def is_in_pos_range(self, x, y, pos, bound):
        return x > pos[0] - bound and y > pos[1] - bound and x < pos[0] + bound and y < pos[1] + bound

    def get_available_pos(self, board):
        reserverd_available_pos = []
        existed_pos = zip(*np.where(board != 2))  # 已经落子的点
        available_pos = []  # 可以落子的点（在主战场附近） [[x1,y1],[x2,y2],...]
        for pos in existed_pos:
            for dir_func in self.directions.values():
                nx, ny = dir_func(*pos)

                if not self.is_outta_range(nx, ny) and board[nx, ny] == 2:
                    if [nx, ny] not in available_pos:
                        available_pos.append([nx, ny])

                    nnx, nny = dir_func(nx, ny)

                    if not self.is_outta_range(nnx, nny) and board[nnx, nny] == 2 and [nnx, nny] not in available_pos:
                        reserverd_available_pos.append(([nx, ny], [nnx, nny]))

        available_pos = list(combinations(available_pos, 2))  # 可以落子的两两组合 [([x1,y1],[x2,y2]), ([x1,y1],[x3,y3]), ...]
        return available_pos + reserverd_available_pos

    def get_available_pos_values(self, board, available_pos, player, board_values, opponent_board_values):
        available_pos_values = []

        for p1, p2 in available_pos:
            total_value = 0
            opponent_total_value = 0
            new_board = board.copy()
            new_board[p1[0], p1[1]] = player
            new_board[p2[0], p2[1]] = player

            for dir_num in range(4):
                board_value = board_values[dir_num].copy()
                opponent_board_value = opponent_board_values[dir_num].copy()

                for p in [p1, p2]:
                    line, line_v = self.get_line(new_board, p, dir_num)
                    self.update_board_value(line, line_v, player, board_value)  # 更新己方价值表
                    self.update_board_value(line, line_v, (player + 1) % 2, opponent_board_value)  # 更新对方价值表

                for i in range(6):
                    c = np.sum(board_value == (i + 1))
                    total_value += c * self.ratio[i]
                    c = np.sum(opponent_board_value == (i + 1))
                    opponent_total_value += c * self.opponent_ratio[i]

            available_pos_values.append(total_value - opponent_total_value)

        return available_pos_values

    def get_board_values(self, board):
        player_board_values = [np.zeros([self.dim, self.dim]), np.zeros([self.dim, self.dim]),
                               np.zeros([self.dim, self.dim]), np.zeros([self.dim, self.dim])]
        opponent_player_board_values = [np.zeros([self.dim, self.dim]), np.zeros([self.dim, self.dim]),
                                        np.zeros([self.dim, self.dim]), np.zeros([self.dim, self.dim])]
        for player in [0, 1]:
            if player == self.player:
                board_values = player_board_values
            else:
                board_values = opponent_player_board_values

            for dir_num in range(2):
                board_value = board_values[dir_num]
                for i in range(self.dim):
                    if dir_num == 0:
                        line, line_v = self.get_line(board, [0, i], dir_num, self.dim)
                    elif dir_num == 1:
                        line, line_v = self.get_line(board, [i, 0], dir_num, self.dim)

                    self.update_board_value(line, line_v, player, board_value)

            for dir_num in range(2, 4):
                board_value = board_values[dir_num]
                for i in range(self.dim):
                    line, line_v = self.get_line(board, [0, i], dir_num, self.dim)

                    self.update_board_value(line, line_v, player, board_value)

                for i in range(1, self.dim):
                    if dir_num == 2:
                        line, line_v = self.get_line(board, [i, 0], dir_num, self.dim)
                    elif dir_num == 3:
                        line, line_v = self.get_line(board, [i, self.dim - 1], dir_num, self.dim)

                    self.update_board_value(line, line_v, player, board_value)

        return player_board_values, opponent_player_board_values

    def choose_action(self, board, last_move=None):
        if np.all(board == 2):  # 下第一步棋
            x, y = [board.shape[0] // 2 + 1], [board.shape[1] // 2 + 1]
            if self.client is not None:
                self.client.move(x, y)
            return x, y

        player_board_values, opponent_player_board_values = self.get_board_values(board)

        available_pos = self.get_available_pos(board)
        available_pos_values = self.get_available_pos_values(board, available_pos, self.player,
                                                             player_board_values, opponent_player_board_values)
        available_pos_opponent_values = self.get_available_pos_values(board, available_pos, (self.player + 1) % 2,
                                                                      opponent_player_board_values, player_board_values)
        
        for vi in range(len(available_pos_values)):
            tmp_values = []
            for i, pos in enumerate(available_pos):
                if available_pos[vi][0] not in pos and available_pos[vi][1] not in pos:
                    tmp_values.append(available_pos_opponent_values[i])

            available_pos_values[vi] -= max(tmp_values)

        pos_value_tuple = list(zip(available_pos, available_pos_values))
        pos_value_tuple.sort(key=lambda t: t[1], reverse=True)

        available_pos, available_pos_values = zip(*pos_value_tuple)
        available_pos, available_pos_values = list(available_pos), list(available_pos_values)

        pos = available_pos[available_pos_values.index(max(available_pos_values))]

        x, y = [pos[0][0], pos[1][0]], [pos[0][1], pos[1][1]]
        if self.client is not None:
            self.client.move(x, y)
        return x, y

    def get_line(self, board, pos, dir_num, bound=6):  # 获取dir_num方向的以pos为中心的坐标
        dir_1, dir_2 = [[1, 0], [0, 1], [1, 1], [1, -1]][dir_num]

        line = []

        ni, nj = pos
        line.append((ni, nj))

        ni, nj = pos[0] - dir_1, pos[1] - dir_2
        while not self.is_outta_range(ni, nj) and self.is_in_pos_range(ni, nj, pos, bound):
            line.insert(0, (ni, nj))
            ni -= dir_1
            nj -= dir_2

        ni, nj = pos[0] + dir_1, pos[1] + dir_2
        while not self.is_outta_range(ni, nj) and self.is_in_pos_range(ni, nj, pos, bound):
            line.append((ni, nj))
            ni += dir_1
            nj += dir_2

        return line, [board[p[0], p[1]] for p in line]

    def update_board_value(self, line, line_v, player, board_value):
        opponent_player = (player + 1) % 2

        for i, _ in enumerate(line_v[:-5]):
            if line_v[i:i + 6].count(opponent_player) == 0:
                board_value[line[i][0], line[i][1]] = line_v[i:i + 6].count(player)
            else:
                board_value[line[i][0], line[i][1]] = 0