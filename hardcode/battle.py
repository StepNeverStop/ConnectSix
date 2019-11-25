
# coding: utf-8
# athor: Keavnn
import random
from pprint import pprint
from absl import app, flags, logging
from absl.flags import FLAGS
from c6 import Connect6
from partial_env import PartialC6

flags.DEFINE_integer('board_size', 19, '棋盘尺寸大小')
flags.DEFINE_integer('box_size', 11, '局部大小')

def main(_argv):
    board_size = FLAGS.board_size
    box_size = FLAGS.box_size
    env = Connect6(dim=board_size, box_size=box_size)
    player1 = RandomPlayer()
    player2 = CounterPlayer()
    battle_loop(env, player1, player2, is_black=True)
    pass


def battle_loop(env, player1, player2=None, is_black=True):
    if player2 is None:
        return
    env.reset()
    if is_black:
        players = [player1, player2]
    else:
        players = [player2, player1]
    x = [0, 0]
    y = [0, 0]
    while True:
        env.render()
        x, y = players[env.current_player].choose_action(env, x, y)
        env.step(x[0], y[0])
        end, winner = env.is_over()
        if end:
            env.render()
            break
        if len(x) == 1 or (x[0] == x[1] and y[0] == y[1]):
            env.step_again()
        else:
            env.step(x[-1], y[-1])
        end, winner = env.is_over()
        if end:
            env.render()
            break


class CounterPlayer:
    def __init__(self, **kwargs):
        pass

    def choose_action(self, env, x, y):
        assert isinstance(x, list)
        assert isinstance(y, list)
        partial_env = PartialC6(env, 1)
        # print(partial_env.available_actions)
        idx0 = partial_env.act()
        print(idx0)
        x0, y0 = idx0 % env.dim, idx0 // env.dim
        print(x0, y0)
        partial_env = PartialC6(env, 0)
        idx1 = partial_env.act()
        if idx1 == idx0:
            idx1 = partial_env.get_next()
        print(idx1)
        x1, y1 = idx1 % env.dim, idx1 // env.dim
        print(x1, y1)
        # input()
        return [x0, x1], [y0, y1]

class RandomPlayer:
    def __init__(self):
        pass

    def choose_action(self, env, *args, **kwargs):
        idx0, idx1 = random.sample(env.available_actions, 2)
        x0, y0 = idx0 % env.dim, idx0 // env.dim
        x1, y1 = idx1 % env.dim, idx1 // env.dim
        if env.move_step == 1:
            return [x0], [y0]
        else:
            return [x0, x1], [y0, y1]


class HumanPlayer:
    def __init__(self):
        pass

    def choose_action(self, *args, **kwargs):
        x, y = [], []
        print('请输入第一个落子点的坐标: ')
        info = input()
        _x, _y = info.split('-')
        x.append(int(_x))
        y.append(int(_y))
        print('请输入第二个落子点的坐标: ')
        info = input()
        _x, _y = info.split('-')
        x.append(int(_x))
        y.append(int(_y))
        return x, y


if __name__ == '__main__':
    try:
        app.run(main)
    except SystemExit:
        pass
