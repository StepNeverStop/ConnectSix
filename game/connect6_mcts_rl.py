import numpy as np
from .connect6 import Connect6


class C6(Connect6):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def reset(self):
        super().reset()
        self.states = {}

    def step(self, x, y):
        """
        执行动作并返回新的棋盘信息
        """
        self.states[x + y * self.dim] = self.current_player
        super().step(x, y)

    def step_again(self):
        assert self.move_step == 1
        self.total_move += 1
        self.moves[self.current_player] += 1
        self.move_step += 1
        if self.move_step == 2:
            self.round += 1
            self.move_step = 0
            self.current_player = (self.current_player + 1) % 2

    def get_state(self):
        """
        用于在执行一个step步后返回新的状态，可以自行定义修改，也可以在AI接收到状态后自行解析成自己希望的输入状态
        """
        return self.board.reshape(-1)

    def get_reward(self):
        """
        用于定义奖励函数
        """
        result = self.is_over()
        if result is not None:
            return 1
        else:
            return 0

    def is_done(self):
        """
        用于判断是否返回done
        """
        result = self.is_over()
        if result is not None:
            return True
        else:
            return False

    def get_current_state(self):
        square_state = np.zeros((4, self.dim, self.dim))
        if self.states:
            moves, players = np.array(list(zip(*self.states.items())))
            move_curr = moves[players == self.current_player]
            move_oppo = moves[players != self.current_player]
            square_state[0][move_curr % self.dim,   # 当前选手之前落过子的点
                            move_curr // self.dim] = 1.0
            square_state[1][move_oppo % self.dim,   # 对手之前落过子的点
                            move_oppo // self.dim] = 1.0

            square_state[2][self.last_move[0],      # 最新的落子点
                            self.last_move[1]] = 1.0
        if self.current_player == 0:
            square_state[3][:, :] = 1.0
        return square_state

    def self_play(self, player):
        self.reset()
        states, probs, players, steps = [], [], [], []
        while True:
            # print(f"--->  第{self.total_move:3d}步")
            x, y, move_probs = player.choose_action(self, return_prob=True, is_self_play=True)
            current_player, move_step = self.get_current_player_info()
            states.append(self.get_current_state())
            probs.append(move_probs)
            players.append(current_player)
            steps.append(move_step)
            # print(self.get_current_state())
            # print(move_probs)
            # print(self.current_player)
            # input()
            self.step(x, y)
            end, winner = self.is_over()
            if end:
                print(f'本次self play共{self.total_move}步.')
                winners_z = np.zeros(len(players))
                if winner != -1:
                    winners_z[np.logical_and(np.array(players) == winner, np.array(steps) == 0)] = -1.0
                    winners_z[np.logical_and(np.array(players) == winner, np.array(steps) == 1)] = 1.0
                    winners_z[np.logical_and(np.array(players) != winner, np.array(steps) == 0)] = 1.0
                    winners_z[np.logical_and(np.array(players) != winner, np.array(steps) == 1)] = -1.0
                player.reset_tree()
                return zip(states, probs, winners_z)

    def play(self, player1, player2):
        self.reset()
        while True:
            if self.current_player == 0:
                x, y = player1.choose_action(self, return_prob=False, is_self_play=False, evaluate=True)
            else:
                x, y = player2.choose_action(self, return_prob=False, is_self_play=False, evaluate=True)
            self.step(x, y)
            end, winner = self.is_over()
            if end:
                if winner == -1:    # 平局
                    return False
                if winner == 0:     # player1胜
                    return True
                if winner == 1:     # player2胜
                    return False
