"""Players used in battle and server modes."""

import random
import time

import requests


class BattleRandomPlayer:
    """Random player that handles Connect6 two-step moves."""

    def choose_action(self, env, *args, **kwargs):
        idx0 = random.sample(env.available_actions, 1)[0]
        idx1 = random.sample(env.available_actions, 1)[0]
        x0, y0 = int(idx0 % env.dim), int(idx0 // env.dim)
        x1, y1 = int(idx1 % env.dim), int(idx1 // env.dim)
        if env.move_step == 1:
            return [x0], [y0]
        return [x0, x1], [y0, y1]


class BattleHumanPlayer:
    """Human player for large-board battle mode."""

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


class RemotePlayer:
    """Player that communicates with a remote game server."""

    def __init__(self, ip='127.0.0.1', port=8080, is_black=True):
        self.url = f'http://{ip}:{port}'
        self.color = 'BLACK' if is_black else 'WHITE'
        response = requests.post(f'{self.url}/login', json={'color': self.color})
        result = response.json()
        if result['code'] == 200:
            self.token = result['token']
            print('注册成功', self.token)
            self.last_x, self.last_y = [], []
        else:
            print('注册失败')

    def move(self, x, y):
        assert len(x) == len(y), 'x与y的长度必须相同'
        movements = []
        for i in range(len(x)):
            movements.append({
                'order': i + 1,
                'color': self.color,
                'x': x[i],
                'y': y[i],
            })
        requests.post(
            f'{self.url}/move',
            headers={'token': self.token},
            json={'movements': movements},
        )

    def choose_action(self):
        while True:
            response = requests.get(f'{self.url}/latest')
            movements = response.json()['movements']

            if len(movements) != 0 and movements[0]['color'] == self.color:
                print('等待对方落子', movements)
                time.sleep(1)
                continue

            tmp_x, tmp_y = [], []
            for movement in movements:
                tmp_x.append(movement['x'])
                tmp_y.append(movement['y'])

            if tmp_x != self.last_x or tmp_y != self.last_y:
                self.last_x, self.last_y = tmp_x, tmp_y
                print('对方落子', movements)
                return tmp_x, tmp_y

            print('等待对方落子', movements)
            time.sleep(1)
