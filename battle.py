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

import requests
import time

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
    def __init__(self, ip='127.0.0.1', port=8080, is_black=True):
        self.url = f'http://{ip}:{port}'
        self.color = 'BLACK' if is_black else 'WHITE'
        r = requests.post(f'{self.url}/login', json={
            'color': self.color
        })
        r = r.json()
        if r['code'] == 200:
            self.token = r['token']
            print('注册成功', self.token)
            self.last_x, self.last_y = [], []
        else:
            print('注册失败')

    def move(self, x, y):  # x: [3, 4] y: [7, 8]
        assert len(x) == len(y), 'x与y的长度必须相同'
        movements = []
        for i in range(len(x)):
            movements.append({
                'order': i + 1,
                'color': self.color,
                'x': x[i],
                'y': y[i]
            })

        requests.post(f'{self.url}/move', headers={
            'token': self.token
        }, json={
            'movements': movements
        })

    def choose_action(self):
        '''
        返回对手走的两步棋
        例如：
            x: [3, 4]
            y: [7, 8]
        如果只走一步，返回: x: [3] y: [4]
        '''
        while True:
            r = requests.get(f'{self.url}/latest')
            movements = r.json()['movements']

            if len(movements) != 0 and movements[0]['color'] == self.color:
                print('等待对方落子', movements)
                time.sleep(1)
                continue

            tmp_x, tmp_y = [], []
            for m in movements:
                tmp_x.append(m['x'])
                tmp_y.append(m['y'])

            if tmp_x != self.last_x or tmp_y != self.last_y:
                self.last_x, self.last_y = tmp_x, tmp_y
                print('对方落子', movements)
                return tmp_x, tmp_y

            print('等待对方落子', movements)
            time.sleep(1)


if __name__ == '__main__':
    try:
        app.run(main)
    except SystemExit:
        pass


# if __name__ == '__main__':
#     import sys
#     is_black = True
#     if len(sys.argv[1:]) != 0:
#         is_black = False
#     print('BLACK' if is_black else 'WHITE')
#     player = TestPlayer(is_black=is_black)
#     if not is_black:
#         x, y = player.choose_action()
#         print('receive', x, y)
#     while True:
#         s = input()
#         x, y = s.split(',')
#         x, y = x.split(' '), y.split(' ')
#         x = [int(i) for i in x]
#         y = [int(i) for i in y]
#         print(x, y)
#         player.move(x, y)
#         x, y = player.choose_action()
#         print('receive', x, y)
