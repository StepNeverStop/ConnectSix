
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
from attack_env import AttackC6

flags.DEFINE_integer('board_size', 37, '棋盘尺寸大小')
flags.DEFINE_integer('box_size', 11, '局部大小')
flags.DEFINE_integer('num', 10, '对局数')
flags.DEFINE_integer('threat', 4, '指定几子相连算威胁，需要围堵')
flags.DEFINE_boolean('render', False, '是否渲染')
flags.DEFINE_string('ip', '58.199.162.110', '指定服务器IP地址')
flags.DEFINE_string('port', '8080', '指定服务器端口号')
flags.DEFINE_enum('color', 'n', ['n', 'b', 'w'],
                  'n: 随机, '
                  'b: 黑子, '
                  'w: 白子')
flags.DEFINE_enum('op', 'random', ['random', 'human', 'self'],
                  'random: 随机策略, '
                  'human: 跟别人对战,'
                  'self: 自我对弈')


class Base:
    def __init__(self):
        pass

    def move(self, *args, **kwargs):
        pass

    def update(self, *args, **kwargs):
        pass


def main(_argv):
    board_size = FLAGS.board_size
    box_size = FLAGS.box_size
    ip = FLAGS.ip
    port = FLAGS.port
    color = FLAGS.color
    op = FLAGS.op
    threat = FLAGS.threat
    env = Connect6(dim=board_size, box_size=box_size)
    count = 0
    tie = 0
    for i in range(FLAGS.num):
        if color == 'n':
            a = random.random()
            if a > 0.5:
                is_black = False    # 极致防御后手
            else:
                is_black = True     # 极致防御先手
        elif color == 'b':
            is_black = True
        else:
            is_black = False

        player1 = CounterPlayer(is_black=is_black, threat=threat)   # 极致防御策略

        if op == 'random':
            player2 = RandomPlayer()
        elif op == 'human':
            if is_black:
                logging.info('自在极意 ----> 黑棋|先手')
            else:
                logging.info('自在极意 ----> 白棋|后手')
            player2 = TestPlayer(ip=ip, port=port, is_black=is_black)
        elif op == 'self':
            player2 = CounterPlayer(is_black=False if is_black else True, threat=threat)
        ret, winner = battle_loop(env, player1, player2, is_black=is_black)
        logging.info(f'第{i:4d}局结束, {ret}')
        if ret:
            count += 1
        elif winner == -1:
            tie += 1
    logging.info(f'对局{FLAGS.num}, 胜场{count}, 平局{tie}, 胜率为: {count/FLAGS.num:.2%}')
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
            input()
        x, y = players[env.current_player].choose_action(env)
        cur_player = env.current_player
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
            players[cur_player].update(env, x[0], y[0])
            env.step_again()
        else:
            env.step(x[-1], y[-1])
            players[cur_player].update(env, x[0], y[0])
            players[cur_player].update(env, x[-1], y[-1])
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


class CounterPlayer(Base):
    def __init__(self, **kwargs):
        super().__init__()
        is_black = kwargs.get('is_black')
        self.threat = kwargs.get('threat', 4)
        if is_black:
            self.flag = 0
        else:
            self.flag = 1
        self.oppo_flag = (self.flag + 1) % 2
        self.attack_action_list = []
        self.attack4_list = []
        self.attack3_list = []
        self.attack2_list = []
        self.win_list = []
        self.actions = []
        self.defence4 = []
        self.defence3 = []
        pass

    def choose_action(self, env):
        random.shuffle(self.win_list)
        # 首先， 能赢肯定要选择可以赢的落子
        while len(self.win_list) > 0:
            a = self.win_list.pop()
            x0, y0 = a[0]
            x1, y1 = a[1]
            if x0 == x1 and y0 == y1:
                continue
            if env.board[y0][x0] == 2 and env.board[y1][x1] == 2:
                return [x0, x1], [y0, y1]

        random.shuffle(self.attack_action_list)
        random.shuffle(self.attack4_list)
        random.shuffle(self.attack3_list)
        random.shuffle(self.attack2_list)
        random.shuffle(self.actions)
        # 其次， 如果对方有连续的四个子，那么肯定需要选择围堵左右两端
        partial_env0 = PartialC6(env, 1, threat=self.threat)
        x0, y0, ergency, list4, list3 = partial_env0.act()
        self.defence4.extend(list4)
        self.defence3.extend(list3)
        low_threat0 = partial_env0.get_low_threat()
        if ergency:  # 如果形势危急
            if env.move_step == 1:  # 而我只能走一步，那么放弃治疗
                return [x0], [y0]
            else:
                x1, y1 = partial_env0.get_next()
                return [x0, x1], [y0, y1]   # 直接选择对对手第一个落子两端围堵
        else:
            partial_env1 = PartialC6(env, 0, threat=self.threat)
            x1, y1, ergency, list4, list3 = partial_env1.act()
            self.defence4.extend(list4)
            self.defence3.extend(list3)
            low_threat1 = partial_env1.get_low_threat()
            if ergency:  # 如果对手第一子不危急，第二子危急
                if env.move_step == 1:  # 而我只能走一步，那么放弃治疗
                    return [x1], [y1]
                else:
                    x2, y2 = partial_env1.get_next()
                    return [x1, x2], [y1, y2]   # 直接选择对对手第二个落子两端围堵
            else:
                # 情况不危机
                ret = [x0, x1], [y0, y1]   # 如果形势不危急，那么我方两子各防守对方一子
        # 然后
        # 如果对方无四子的， 那么我方选择两步进攻的棋子，尽可能凑成四个子
        # 如果对方有一步为四子的，且一端已经被堵住，那么我方将围堵另一端，并选择一步进攻性落子，尽可能形成四子
        # 如果对方两步都为四子，且都是一端被堵住，那么我方两次落子应该都是围堵
        if env.move_step == 1:
            if low_threat0 and low_threat1:
                while len(self.attack4_list) > 0:
                    x, y = self.attack4_list.pop()
                    if env.judge(x, y, self.flag, self.oppo_flag):
                        return [x], [y]
            elif low_threat0:
                return [x1], [y1]
            elif low_threat1:
                return [x0], [y0]
            return [x0], [y0]
        else:
            if low_threat0 and low_threat1:
                _r = self.attack2(env)
                if _r is not None:
                    return _r
                while len(self.attack_action_list) > 0:  # 进攻两步 形成四子
                    a = self.attack_action_list.pop()
                    _x0, _y0 = a[0]
                    _x1, _y1 = a[1]
                    if _x0 == _x1 and _y0 == _y1:
                        continue
                    if env.board[_y0][_x0] == 2 and env.board[_y1][_x1] == 2:
                        k = 1
                        __i = 1
                        __j = 1
                        if _x0 == _x1:
                            for i in range(1, 6):
                                if 0 <= _y0 + i < env.dim:
                                    if env.board[_y0 + i][_x0] != self.oppo_flag:
                                        k += __i
                                    else:
                                        __i = 0
                                if 0 <= _y0 - i < env.dim:
                                    if env.board[_y0 - i][_x0] != self.oppo_flag:
                                        k += __j
                                    else:
                                        __j = 0
                                if k >= 6:
                                    break
                        elif _y0 == _y1:
                            for i in range(1, 6):
                                if 0 <= _x0 + i < env.dim:
                                    if env.board[_y0][_x0 + i] != self.oppo_flag:
                                        k += __i
                                    else:
                                        __i = 0
                                if 0 <= _x0 - i < env.dim:
                                    if env.board[_y0][_x0 - i] != self.oppo_flag:
                                        k += __j
                                    else:
                                        __j = 0
                                if k >= 6:
                                    break
                        elif _y0 - _y1 == _x0 - _x1:
                            for i in range(1, 6):
                                if 0 <= _x0 + i < env.dim and 0 <= _y0 + i < env.dim:
                                    if env.board[_y0 + i][_x0 + i] != self.oppo_flag:
                                        k += __i
                                    else:
                                        __i = 0
                                if 0 <= _x0 - i < env.dim and 0 <= _y0 - i < env.dim:
                                    if env.board[_y0 - i][_x0 - i] != self.oppo_flag:
                                        k += __j
                                    else:
                                        __j = 0
                                if k >= 6:
                                    break
                        else:
                            for i in range(1, 6):
                                if 0 <= _x0 + i < env.dim and 0 <= _y0 - i < env.dim:
                                    if env.board[_y0 - i][_x0 + i] != self.oppo_flag:
                                        k += __i
                                    else:
                                        __i = 0
                                if 0 <= _x0 - i < env.dim and 0 <= _y0 + i < env.dim:
                                    if env.board[_y0 + i][_x0 - i] != self.oppo_flag:
                                        k += __j
                                    else:
                                        __j = 0
                                if k >= 6:
                                    break
                        if k >= 6:
                            return [_x0, _x1], [_y0, _y1]
                for a in self.actions:
                    act3 = env.get3(a[0], a[1], self.oppo_flag)
                    if act3 is not None:
                        return list(zip(*act3))
                    else:
                        self.actions.remove(a)
            elif low_threat0:
                _r = self.defence1attack1(env, x1, y1)
                if _r is not None:
                    return _r
            elif low_threat1:
                _r = self.defence1attack1(env, x0, y0)
                if _r is not None:
                    return _r
            else:
                [x0, x1], [y0, y1] = ret
                if x0 == x1 and y0 == y1:
                    while len(self.attack4_list) > 0:
                        x, y = self.attack4_list.pop()
                        if x == x0 and y ==y0:
                            continue
                        if env.judge(x, y, self.flag, self.oppo_flag, 3):
                            return [x0, x], [y0, y]
                    while len(self.attack3_list) > 0:
                        x, y = self.attack3_list.pop()
                        if x == x0 and y ==y0:
                            continue
                        if env.judge(x, y, self.flag, self.oppo_flag, 2):
                            return [x0, x], [y0, y]
                    while len(self.attack2_list) > 0:
                        x, y = self.attack2_list.pop()
                        if x == x0 and y ==y0:
                            continue
                        if env.judge(x, y, self.flag, self.oppo_flag, 1):
                            return [x0, x], [y0, y]
                    x1, y1 = partial_env1.get_next()
                return [x0, x1], [y0, y1]
            return ret

    def update(self, env, x, y):
        self.actions.append([x, y])
        _env = AttackC6(env, x, y, self.flag)
        aaaa = _env.get_actions()
        self.attack_action_list.extend(aaaa[0])
        self.win_list.extend(aaaa[1])
        self.attack4_list.extend(_env.sigle(3))
        self.attack3_list.extend(_env.sigle(2))
        self.attack2_list.extend(_env.sigle(1))

    def defence1attack1(self, env, xx, yy):
        while len(self.defence4) > 0:
            x, y = self.defence4.pop()
            if x == xx and y ==yy:
                continue
            if env.board[y][x] == 2:
                return [xx, x], [yy, y]
        while len(self.attack4_list) > 0:
            x, y = self.attack4_list.pop()
            if x == xx and y ==yy:
                continue
            if env.judge(x, y, self.flag, self.oppo_flag, 3):
                return [xx, x], [yy, y]
        while len(self.defence3) > 0:
            x, y = self.defence3.pop()
            if x == xx and y ==yy:
                continue
            if env.board[y][x] == 2:
                return [xx, x], [yy, y]
        while len(self.attack3_list) > 0:
            x, y = self.attack3_list.pop()
            if x == xx and y ==yy:
                continue
            if env.judge(x, y, self.flag, self.oppo_flag, 2):
                return [xx, x], [yy, y]
        while len(self.attack2_list) > 0:
            x, y = self.attack2_list.pop()
            if x == xx and y ==yy:
                continue
            if env.judge(x, y, self.flag, self.oppo_flag, 1):
                return [xx, x], [yy, y]
        return None
    
    def attack2(self, env):
        x0, y0 = -1, -1
        for i in self.attack4_list:
            x, y = i
            if env.judge(x, y, self.flag, self.oppo_flag, 3):
                x0=x
                y0=y
                break
        if x0 != -1:
            for i in self.defence4:
                x, y = i
                if x == x0 and y ==y0:
                    continue
                if env.board[y][x] == 2:
                    return [x,x0],[y,y0]
            for i in self.attack4_list:
                x, y = i
                if x == x0 and y ==y0:
                    continue
                if env.judge(x, y, self.flag, self.oppo_flag, 3):
                    return [x,x0],[y,y0]
            for i in self.defence3:
                x, y = i
                if x == x0 and y ==y0:
                    continue
                if env.board[y][x] == 2:
                    return [x,x0],[y,y0]
            for i in self.attack3_list:
                x, y = i
                if x == x0 and y ==y0:
                    continue
                if env.judge(x, y, self.flag, self.oppo_flag, 2):
                    return [x,x0],[y,y0]
            for i in self.attack2_list:
                x, y = i
                if x == x0 and y ==y0:
                    continue
                if env.judge(x, y, self.flag, self.oppo_flag, 1):
                    return [x,x0],[y,y0]
        return None





class RandomPlayer(Base):
    def __init__(self):
        super().__init__()
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


class HumanPlayer(Base):
    def __init__(self):
        super().__init__()
        pass

    def choose_action(self, env, *args, **kwargs):
        x, y = [], []
        logging.info('请输入第一个落子点的坐标: ')
        info = input()
        _x, _y = info.split('-')
        x.append(int(_x))
        y.append(int(_y))
        if env.move_step == 1:
            return x, y
        else:
            logging.info('请输入第二个落子点的坐标: ')
            info = input()
            _x, _y = info.split('-')
            x.append(int(_x))
            y.append(int(_y))
            return x, y

    def move(self, *args, **kwargs):
        pass


class TestPlayer(Base):
    def __init__(self, ip='127.0.0.1', port=8080, is_black=True):
        super().__init__()
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
