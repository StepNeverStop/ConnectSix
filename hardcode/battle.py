
# coding: utf-8
# athor: Keavnn
import random
import requests
import time
from pprint import pprint
from absl import app, flags, logging
from absl.flags import FLAGS
from c6 import Connect6
from partial_env import PartialC6

flags.DEFINE_integer('board_size', 37, '棋盘尺寸大小')
flags.DEFINE_integer('box_size', 11, '局部大小')
flags.DEFINE_integer('num', 10, '对局数')
flags.DEFINE_boolean('render', False, '是否渲染')


def main(_argv):
    board_size = FLAGS.board_size
    box_size = FLAGS.box_size
    env = Connect6(dim=board_size, box_size=box_size)
    player1 = CounterPlayer()   # 极致防御策略
    player2 = RandomPlayer()
    count = 0
    tie = 0
    for i in range(FLAGS.num):
        a = random.random()
        if a > 0.5:
            is_black = False    # 极致防御后手
        else:
            is_black = True     # 极致防御先手
        # print(is_black)
        # player2 = TestPlayer(ip='58.199.162.110', port=8080, is_black=is_black)
        ret, winner = battle_loop(env, player1, player2, is_black=True)
        print(f'第{i}局结束, {ret}')
        if ret:
            count += 1
        elif winner == -1:
            tie += 1
    print(f'对局{FLAGS.num}, 胜场{count}, 平局{tie}, 胜率为: {count/FLAGS.num:.2%}')
    pass


def battle_loop(env, player1, player2=None, is_black=True):
    if player2 is None:
        return
    env.reset()
    if is_black:
        players = [player1, player2]
    else:
        players = [player2, player1]
    while True:
        if FLAGS.render:
            env.render()
        x, y = players[env.current_player].choose_action(env)
        # ---
        if is_black:
            if env.current_player == 0:
                players[-1].move(x, y)
        else:
            if env.current_player == 1:
                players[0].move(x, y)
        # ---
        env.step(x[0], y[0])
        end, winner = env.is_over()
        if end:
            if FLAGS.render:
                env.render()
            break
        if len(x) == 1 or (x[0] == x[1] and y[0] == y[1]):
            env.step_again()
        else:
            env.step(x[-1], y[-1])
        end, winner = env.is_over()
        if end:
            if FLAGS.render:
                env.render()
            break
    if is_black:
        if winner == 0:
            return True, winner
        else:
            return False, winner
    else:
        if winner == 0:
            return False, winner
        else:
            return True, winner


class CounterPlayer:
    def __init__(self, **kwargs):
        pass

    def choose_action(self, env):
        partial_env = PartialC6(env, 1)
        idx0, ergency = partial_env.act()
        x0, y0 = int(idx0 % env.dim), int(idx0 // env.dim)
        if env.move_step == 1:
            # print(idx0)
            # print(x0, y0)
            return [x0], [y0]
        else:
            if ergency:
                idx1 = partial_env.get_next()
                x1, y1 = int(idx1 % env.dim), int(idx1 // env.dim)
                return [x0, x1], [y0, y1]

            partial_env = PartialC6(env, 0)
            idx1, ergency = partial_env.act()
            if idx1 == idx0:
                idx1 = partial_env.get_next()
            x1, y1 = int(idx1 % env.dim), int(idx1 // env.dim)
            return [x0, x1], [y0, y1]
            # print(idx0, idx1)
            # print(x0, y0)
            # print(x1, y1)

    def move(self, *args, **kwargs):
        pass


class RandomPlayer:
    def __init__(self):
        pass

    def choose_action(self, env, *args, **kwargs):
        idx0 = random.sample(env.available_actions, 1)[0]
        idx1 = random.sample(env.available_actions, 1)[0]
        x0, y0 = int(idx0 % env.dim), int(idx0 // env.dim)
        x1, y1 = int(idx1 % env.dim), int(idx1 // env.dim)
        if env.move_step == 1:
            return [x0], [y0]
        else:
            return [x0, x1], [y0, y1]

    def move(self, *args, **kwargs):
        pass


class HumanPlayer:
    def __init__(self):
        pass

    def choose_action(self, env, *args, **kwargs):
        x, y = [], []
        print('请输入第一个落子点的坐标: ')
        info = input()
        _x, _y = info.split('-')
        x.append(int(_x))
        y.append(int(_y))
        if env.move_step == 1:
            return x, y
        else:
            print('请输入第二个落子点的坐标: ')
            info = input()
            _x, _y = info.split('-')
            x.append(int(_x))
            y.append(int(_y))
            return x, y

    def move(self, *args, **kwargs):
        pass


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

    def choose_action(self, *args, **kwargs):
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
