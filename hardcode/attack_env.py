import numpy as np
import random


def reverse_of(dir_func):
    '''
    反转搜索方向
    '''
    dx, dy = dir_func(0, 0)  # differentiate
    return lambda x, y: (x - dx, y - dy)


class AttackC6(object):
    def __init__(self, env, x, y, flag):
        self.env = env
        self.x = x
        self.y = y
        self.flag = flag
        self.oppo_flag = (flag + 1) % 2
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
        self.board, self.xx, self.yy = env.get_attack_partial_env(x, y)
        # self.board[self.yy, self.xx] = self.flag
        self.diff_x = self.x - self.xx
        self.diff_y = self.y - self.yy

    def is_outta_range(self, x, y):
        return x < 0 or x >= self.dim or y < 0 or y >= self.dim

    def sigle(self, num=3):
        a = []
        i = 0
        for _dir, dir_func in self.directions.items():
            i += 1
            if i > 4:
                break
            b = []
            count = 1
            nx0, ny0 = dir_func(self.xx, self.yy)
            if not self.is_outta_range(nx0, ny0):  # 判断x y是不是边界点
                while self.board[ny0][nx0] == self.flag:
                    count += 1
                    nx0, ny0 = dir_func(nx0, ny0)
                    if self.is_outta_range(nx0, ny0):
                        break
                if not self.is_outta_range(nx0, ny0) and self.board[ny0][nx0] == 2:
                    b.append([nx0, ny0])

            reverse_dir_func = reverse_of(dir_func)
            nx1, ny1 = reverse_dir_func(self.xx, self.yy)
            if not self.is_outta_range(nx1, ny1):  # 判断x y是不是边界点
                while self.board[ny1][nx1] == self.flag:
                    count += 1
                    nx1, ny1 = reverse_dir_func(nx1, ny1)
                    if self.is_outta_range(nx1, ny1):
                        break
                if not self.is_outta_range(nx1, ny1) and self.board[ny1][nx1] == 2:
                    b.append([nx1, ny1])
            if count >= num:
                a.extend(b)
        if len(a) != 0:
            a = (np.array(a) + np.array([self.diff_x, self.diff_y])).tolist()
        return a

    def get_actions(self):
        al = []  # 连456个
        bl = []  # 连6个
        shuf = [0, 1, 2, 3]
        random.shuffle(shuf)
        for k in shuf:
            if k == 0:
                # right
                for i in range(6):
                    part = self.board[self.yy, i:i + 6]
                    if (part == self.oppo_flag).any():
                        continue
                    if np.where(part == self.flag)[0].shape[0] >= 2:
                        t = np.where(part == 2)[0] + i
                        v = np.where(part != 2)[0] + i
                        ll = self.sep_actions(t, v, self.yy, axis='x')
                        if len(t) > 2:
                            al.extend(ll)
                        else:
                            bl.extend(ll)
            elif k == 1:
                # up
                for i in range(6):
                    part = self.board[i:i + 6, self.xx]
                    if (part == self.oppo_flag).any():
                        continue
                    if np.where(part == self.flag)[0].shape[0] >= 2:
                        t = np.where(part == 2)[0] + i
                        v = np.where(part != 2)[0] + i
                        ll = self.sep_actions(t, v, self.xx, axis='y')
                        if len(t) > 2:
                            al.extend(ll)
                        else:
                            bl.extend(ll)
            elif k == 2:
                # left top 2 right bottom
                _diff = self.xx - self.yy
                if -6 < _diff < 6:
                    diag = self.board.diagonal(_diff)
                    for i in range(6 - abs(_diff)):
                        part = diag[i:i + 6]
                        if (part == self.oppo_flag).any():
                            continue
                        if np.where(part == self.flag)[0].shape[0] >= 2:
                            t = np.where(part == 2)[0] + i
                            v = np.where(part != 2)[0] + i
                            ll = self.sep_actions(t, v, -1, axis='l2r')
                            if len(t) > 2:
                                al.extend(ll)
                            else:
                                bl.extend(ll)
            elif k == 3:
                # right top 2 left bottom
                _diff = 10 - self.xx - self.yy
                if -6 < _diff < 6:
                    diag = np.fliplr(self.board).diagonal(10 - self.xx - self.yy)
                    for i in range(6 - abs(_diff)):
                        part = diag[i:i + 6]
                        if (part == self.oppo_flag).any():
                            continue
                        if np.where(part == self.flag)[0].shape[0] >= 2:
                            t = np.where(part == 2)[0] + i  # 空位
                            v = np.where(part != 2)[0] + i  # 已落子己方位置
                            ll = self.sep_actions(t, v, -1, axis='r2l')
                            if len(t) > 2:
                                al.extend(ll)
                            else:
                                bl.extend(ll)

        if len(al) != 0:
            al = (np.array(al) + np.array([self.diff_x, self.diff_y])).tolist()
        if len(bl) != 0:
            bl = (np.array(bl) + np.array([self.diff_x, self.diff_y])).tolist()
        return al, bl

    def sep_actions(self, t, v, xy, axis):
        a = []
        if len(v) == 2:
            if abs(v[0] - v[1]) == 1:
                _min = min(v)
                _max = max(v)
                for i in t:
                    for j in t:
                        if i == j:
                            continue
                        if (i == _min - 1 and j == _min - 2) or (i == _max + 1 and j == _max + 2):
                            a.append([i, j])
            if abs(v[0] - v[1]) == 2:
                _min = min(v) - 1
                _max = max(v) + 1
                for i in t:
                    for j in t:
                        if i == j:
                            continue
                        if _min <= i <= _max and _min <= j <= _max and abs(i - j) <= 2:
                            a.append([i, j])
            elif abs(v[0] - v[1]) == 3:
                _min = min(v)
                _max = max(v)
                a.append([_min + 1, _max - 1])
        elif len(v) == 4:
            a.append([t[0], t[1]])

        b = []
        if axis == 'x':
            for i in a:
                b.append(
                    [[i[0], xy], [i[1], xy]]
                )
        elif axis == 'y':
            for i in a:
                b.append(
                    [[xy, i[0]], [xy, i[1]]]
                )
        elif axis == 'l2r':
            for i in a:
                b.append(
                    [[i[0], i[0]], [i[1], i[1]]]
                )
        elif axis == 'r2l':
            for i in a:
                b.append(
                    [[10 - i[0], i[0]], [10 - i[1], i[1]]]
                )
        return b
