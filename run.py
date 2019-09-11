import sys
import time
import numpy as np
from game import Connect6
from bot import RandomBot, Player
from policy import MyBot

if __name__ == '__main__':
    print('Welcome to Keavnn\'s Connect6.')
    print('Choose player slot. (1=Player 2=AI)')
    BOARD_DIMENSION = 19

    info = ['黑子', '白子']
    player1_choice, player2_choice = '2', '2'
    player1_name, player2_name = 'wjs', 'zzy'
    # player1_choice, player1_name = input(' Player1 (1 or 2) : '), input(' Player1 name: ')
    # player2_choice, player2_name = input(' Player2 (1 or 2) : '), input(' Player2 name: ')
    p1 = Player(BOARD_DIMENSION) if player1_choice == '1' else MyBot(BOARD_DIMENSION, 'black')
    time.sleep(1)
    p2 = Player(BOARD_DIMENSION) if player2_choice == '1' else MyBot(BOARD_DIMENSION, 'white')
    bots_name = [player1_name, player2_name]
    bots = [p1, p2]
    wins = [0, 0]

    env = Connect6(BOARD_DIMENSION)
    for episode in range(10000):
        if np.random.rand() > 0.5:
            bots_name.reverse()
            bots.reverse()
            wins.reverse()
        env.register(bots_name[0], bots_name[1])
        init_state = env.reset()
        state = init_state

        player = 0
        move_step = 1
        total_step = 0
        states = [
            [init_state, init_state],  # 黑棋状态
            [init_state, init_state],  # 白棋状态
        ]

        while True:
            total_step += 1
            # env.render()
            while True:  # 如果选择的动作不合法，就重新选择。这个一般只对随机智能体或者手动输入时使用，实现自己的智能体时可以根据需求计算出每一步的可行动空间，避免重新选择动作
                try:
                    x, y = bots[player].choose_action(state)
                    # print(f'{bots_name[player]}选择, x: {x+1}, y: {y+1}')
                    is_ok, msg = env.can_place(x, y)
                    if not is_ok:
                        # print(f'{msg}, 请换个位置')
                        continue
                    break
                except KeyboardInterrupt:
                    print('游戏中断')
                    sys.exit()

            state = env.step(x, y)
            states[player][-1] = state
            move_step += 1
            if wins[player] - wins[(player + 1)%2] < 6:
                bots[player].learn()    # 如果想让AI每两步学习一次，可以将其放置在下方判断中

            result = env.is_over()
            if result is not None:
                if result == -1:
                    r0, r1= 0.1, 0.1
                    print(f'episode: {episode}, step: {total_step:>3d}, 平局')
                else:
                    r0, r1 = 1, -move_step
                    print(f'episode: {episode:>4d}, step: {total_step:>3d}, {info[player]}{bots_name[player]}获胜')
                    wins[player] += 1

                bots[player].store(s=states[player][0], r=r0, s_=states[player][1], done=True)
                bots[player].writer_loop_summary(episode, reward=r0, step=total_step)
                op = (player + 1) % 2
                bots[op].store(s=states[op][0], r=r1, s_=states[op][1], done=True)
                bots[op].writer_loop_summary(episode, reward=r1, step=total_step)
                # env.render()
                # print(env.last_move + 1)
                # print(f'游戏结束，{bots_name[player]}获胜')
                break

            if move_step == 2:
                move_step = 0
                player = (player + 1) % 2
                bots[player].store(s=states[player][0], r=0, s_=states[player][1], done=False)
                states[player][0] = states[(player + 1) % 2][1]
