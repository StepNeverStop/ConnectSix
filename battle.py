# coding: utf-8
# athor: Keavnn
import numpy as np
from pprint import pprint
from absl import app, flags, logging
from absl.flags import FLAGS
from game.connect6_mcts_rl import C6
from player.mcts_rl import MCTS_POLICY, MCTSRL
from utils.timer import timer
from utils.sth import load_config

flags.DEFINE_integer('board_size', 37, '棋盘尺寸大小')
flags.DEFINE_integer('box_size', 19, 'box的大小')
flags.DEFINE_string('model_path', './models', '指定要加载的模型文件夹')


def main(_argv):
    config = load_config('./battle_config.yaml')
    board_size = FLAGS.board_size
    box_size = FLAGS.box_size
    model_path = FLAGS.model_path + '/models' + str(FLAGS.box_size)
    pprint(config)
    env = C6(
        dim=board_size,
        box_size=box_size
    )
    net = MCTS_POLICY(
        state_dim=[box_size, box_size, 4]
    )
    net.restore(model_path)
    player = MCTSRL(
        pv_net=net,
        temp=config['temp'],
        c_puct=config['c_puct'],
        playout_num=config['playout_num'],
        dim=box_size,
        name='mcts_rl_policy'
    )
    player2 = HumanPlayer(board_size)
    battle_loop(env, player, player2, is_black=True)


def battle_loop(env, player1, player2=None, is_black=True):
    if player2 is None:
        return
    env.reset()
    if is_black:
        players = [player1, player2]
    else:
        players = [player2, player1]
    while True:
        x, y = players[env.current_player].choose_action(env, return_prob=False, is_self_play=False)
        print(x, y)
        input()
        if isinstance(x, list):
            env.step(x[0], y[0])
            if len(x) == 1:
                env.step_again()
            else:
                env.step(x[-1], y[-1])
        else:
            env.step(x, y)
        env.render()
        end, winner = env.is_over()
        if end:
            break


class HumanPlayer:
    def __init__(self, board_size):
        self.board_size = board_size

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


class TestPlayer:
    def __init__(self, *args, **kwargs):
        pass

    def choose_action(*args, **kwargs):
        '''
        返回对手走的两步棋
        例如：
            x: [3, 4]
            y: [7, 8]
        如果只走一步，返回: x: [3] y: [4]
        '''
        return [0], [0]


if __name__ == '__main__':
    try:
        app.run(main)
    except SystemExit:
        pass
