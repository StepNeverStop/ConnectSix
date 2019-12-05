import time
import random

import requests
import numpy as np

from connect6 import Connect6
from zzy_bot import ZZY_Bot


def print_matrix(board):
    b = board.astype(np.str)
    print('     ', end='')
    for j in range(board.shape[1]):
        print(f'{j:3}| ', end='')

    print()
    for i in range(board.shape[0]):
        print(f'{i:3}| ', end='')
        for j in range(board.shape[1]):
            t = b[i, j] if b[i, j] != '2' and b[i, j] != '0.0' else ' '
            print(f'{t:3}| ', end='')
        print()


class RandomBot():
    def __init__(self, client):
        self.client = client

    def choose_action(self, board, last_move):
        x, y = [], []
        n = 2
        if np.all(board == 2):
            n = 1

        for i in range(n):
            x.append(random.randrange(0, board.shape[0]))
            y.append(random.randrange(0, board.shape[1]))

        self.client.move(x, y)
        return x, y


class Simulated_Player:
    def __init__(self, client):
        self.client = client

    def choose_action(self, board, last_move):
        x, y = self.client.wait_for_opponent_action(board, last_move)
        return x, y


class Client:
    def __init__(self, ip='127.0.0.1', port=8080, is_black=True):
        self.url = f'http://{ip}:{port}'
        self.color = 'BLACK' if is_black else 'WHITE'
        print(self.color)
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
                'x': int(x[i]),
                'y': int(y[i])
            })

        requests.post(f'{self.url}/move', headers={
            'token': self.token
        }, json={
            'movements': movements
        })

    def wait_for_opponent_action(self, board, last_move):
        '''
        返回对手走的两步棋
        例如：
            x: [3, 4]
            y: [7, 8]
        如果只走一步，返回: x: [3] y: [4]
        '''
        if not np.all(board == 2):
            self.move(*last_move)

        while True:
            r = requests.get(f'{self.url}/latest')
            movements = r.json()['movements']

            if len(movements) != 0 and movements[0]['color'] == self.color:
                print('等待落子...')
                time.sleep(1)
                continue

            tmp_x, tmp_y = [], []
            for m in movements:
                tmp_x.append(m['x'])
                tmp_y.append(m['y'])

            if tmp_x != self.last_x or tmp_y != self.last_y:
                self.last_x, self.last_y = tmp_x, tmp_y
                return tmp_x, tmp_y

            print('等待落子...')
            time.sleep(1)


if __name__ == "__main__":
    from zzy_bot_1 import ZZY_Bot_1

    client_b = Client(is_black=True)
    client_w = Client(is_black=False)
    env = Connect6(37)

    players = [ZZY_Bot(client_b, 37, 0), ZZY_Bot_1(client_w, 37, 1)]

    while True:
        print(env.current_player, '落子')
        board, last_move = env.get_current_state()
        _time = time.time()
        x, y = players[env.current_player].choose_action(board, last_move)
        time.sleep(0.5)
        print(env.current_player, time.time() - _time)
        env.step(x, y)
        print(x, y)
        # print_matrix(env.board)

        end, winner = env.is_over()
        if end:
            print(f'winner {winner}')
            break
    ############################################
    # from zzy_bot_1 import ZZY_Bot_1
    # env = Connect6(37)

    # players = [ZZY_Bot(None, 37, 0), ZZY_Bot_1(None, 37, 1)]

    # while True:
    #     print(env.current_player, '落子')
    #     board, last_move = env.get_current_state()
    #     x, y = players[env.current_player].choose_action(board, last_move)
    #     time.sleep(0.5)
    #     env.step(x, y)
    #     print(x, y)
    #     # print_matrix(env.board)

    #     end, winner = env.is_over()
    #     if end:
    #         print(f'winner {winner}')
    #         break

    #############################

    # is_black = False

    # client = Client(is_black=is_black)

    # env = Connect6(37)

    # if is_black:
    #     players = [ZZY_Bot(client, 37, 0), Simulated_Player(client)]
    # else:
    #     players = [Simulated_Player(client), ZZY_Bot(client, 37, 1)]

    # while True:
    #     print(env.current_player, '落子')
    #     board, last_move = env.get_current_state()
    #     x, y = players[env.current_player].choose_action(board, last_move)
    #     print(x, y)
    #     env.step(x, y)

    #     end, winner = env.is_over()
    #     if end:
    #         print(f'winner {winner}')
    #         break

    ######################

    # bot = ZZY_Bot(None, 37, 0)
    # board = np.full([37, 37], 2, dtype=np.int32)
    # board[19, 19]=0
    # board[19, 20]=1
    # board[20, 19]=1
    # board[20, 18]=0
    # board[21, 18]=0
    # board[18, 21]=1
    # board[17, 22]=1
    # board[16, 23]=0
    # board[21, 19]=0
    # board[16, 21]=1
    # board[17, 20]=1
    # board[17, 21]=0
    # board[18, 20]=0
    # board[16, 22]=1
    # board[21, 17]=1
    # board[18, 18]=0
    # board[19, 18]=0
    # board[17, 18]=1
    # board[22, 18]=1
    # board[15, 22]=0
    # board[17, 19]=0
    # board[16, 19]=1
    # board[19, 22]=1
    # board[15, 18]=0
    # board[20, 23]=0
    # board[16, 17]=1
    # board[18, 19]=1
    # print_matrix(board)
    # x, y = bot.choose_action(board)
    # print(x, y)
    # new_board = board.copy().astype(np.str)
    # new_board[x[0], y[0]] = 'X'
    # new_board[x[1], y[1]] = 'X'
    # print_matrix(new_board)
