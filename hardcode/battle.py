
# coding: utf-8
# athor: Keavnn
from pprint import pprint
from absl import app, flags, logging
from absl.flags import FLAGS
from c6 import Connect6
from partial_env import PartialC6



def main(_argv):
    board_size = FLAGS.board_size
    env = Connect6(dim=19)
    player1 = HumanPlayer()
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
        if len(x) == 1 or (x[0] == x[1] and y[0] == y[1]):
            env.step_again()
        else:
            env.step(x[-1], y[-1])
        end, winner = env.is_over()
        if end:
            break


class CounterPlayer:
    def __init__(self, **kwargs):
        pass

    def choose_action(self, env, x, y):
        assert isinstance(x, list)
        assert isinstance(y, list)
        l = len(x)
        if l == 1:
            partial_env = PartialC6(env, 11, 0)
            pass
        elif l == 2:
            partial_env = PartialC6(env, 11, 0)
            x, y = partial_env.act()
            partial_env = PartialC6(env, 11, 1)


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
