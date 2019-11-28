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
        self.board[self.yy, self.xx] = self.flag
        self.diff_x = self.x - self.xx
        self.diff_y = self.y - self.yy

    def is_outta_range(self, x, y):
        return x < 0 or x >= self.dim or y < 0 or y >= self.dim

    def get_actions(self):
        al = []  # 连456个
        bl = []  # 连6个
        # right
        for i in range(6):
            part = self.board[self.yy, i:i + 6]
            if (part == self.oppo_flag).any():
                continue
            if np.where(part == self.flag)[0].shape[0] >= 2:
                t = np.where(part == 2)[0]
                l = t + i
                ll = self.sep_actions(l, self.yy, axis='x')
                if len(t) > 2:
                    al.extend(ll)
                else:
                    bl.extend(ll)

        # up
        for i in range(6):
            part = self.board[i:i + 6, self.xx]
            if (part == self.oppo_flag).any():
                continue
            if np.where(part == self.flag)[0].shape[0] >= 2:
                t = np.where(part == 2)[0]
                l = t + i
                ll = self.sep_actions(l, self.xx, axis='y')
                if len(t) > 2:
                    al.extend(ll)
                else:
                    bl.extend(ll)

        # left top 2 right bottom
        _diff = self.xx - self.yy
        if -6 < _diff < 6:
            diag = self.board.diagonal(_diff)
            for i in range(6 - abs(_diff)):
                part = self.board[i:i + 6]
                if (part == self.oppo_flag).any():
                    continue
                if np.where(part == self.flag)[0].shape[0] >= 2:
                    t = np.where(part == 2)[0]
                    l = t + i
                    ll = self.sep_actions(l, -1, axis='l2r')
                    if len(t) > 2:
                        al.extend(ll)
                    else:
                        bl.extend(ll)
        # right top 2 left bottom
        _diff = 10 - self.xx - self.yy
        if -6 < _diff < 6:
            diag = np.fliplr(self.board).diagonal(10 - self.xx - self.yy)
            for i in range(6 - abs(_diff)):
                part = self.board[i:i + 6]
                if (part == self.oppo_flag).any():
                    continue
                if np.where(part == self.flag)[0].shape[0] >= 2:
                    t = np.where(part == 2)[0]
                    l = t + i
                    ll = self.sep_actions(l, -1, axis='r2l')
                    if len(t) > 2:
                        al.extend(ll)
                    else:
                        bl.extend(ll)

        if len(al) != 0:
            al = (np.array(al) + np.array([self.diff_x, self.diff_y])).tolist()
        if len(bl) != 0:
            bl = (np.array(bl) + np.array([self.diff_x, self.diff_y])).tolist()
        return al, bl

    def sep_actions(self, l, xy, axis):
        a = []
        for i in l:
            for j in l:
                if i == j:
                    continue
                a.append([i, j])
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
