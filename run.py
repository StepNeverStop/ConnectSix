from game import Connect6
from bot import RandomBot, Player

if __name__ == '__main__':
    print('Welcome to Keavnn\'s Connect6.')
    print('Choose player slot. (1=Player 2=AI)')
    BOARD_DIMENSION = 19
    black_choice, white_choice = '2', '2'
    black_name, white_name = '黑棋wjs', '白棋zzy'
    # black_choice, black_name = input(' Black (1 or 2) : '), input(' Black name: ')
    # white_choice, white_name = input(' White (1 or 2) : '), input(' White name: ')
    blackbot = Player(BOARD_DIMENSION) if black_choice == '1' else RandomBot(BOARD_DIMENSION)
    whitebot = Player(BOARD_DIMENSION) if white_choice == '1' else RandomBot(BOARD_DIMENSION)
    bots_name = [black_name, white_name]
    bots = [blackbot, whitebot]

    env = Connect6(BOARD_DIMENSION, black_name, white_name)
    state = env.reset()

    player = 0
    move_step = 1
    states = [
        [state, None],  # 黑棋状态
        [None, None],  # 白棋状态
    ]

    while True:
        # env.render()
        while True: # 如果选择的动作不合法，就重新选择。这个一般只对随机智能体或者手动输入时使用，实现自己的智能体时可以根据需求计算出每一步的可行动空间，避免重新选择动作
            try:
                x, y = bots[player].choose_action(state)
                print(f'{bots_name[player]}选择, x: {x+1}, y: {y+1}')
                is_ok, msg = env.can_place(x, y)
                if not is_ok:
                    print(f'{msg}, 请换个位置')
                    continue
                break
            except KeyboardInterrupt:
                print('游戏中断')
                sys.exit()
        states[player][-1] = env.step(x, y)
        move_step += 1
        bots[player].learn()    # 如果想让AI每两步学习一次，可以将其放置在下方判断中

        if env.is_over() is not None:
            bots[player].store(
                s=states[player][0],
                r=1,
                s_=states[player][1],
                done=True
            )
            op = (player + 1) % 2
            bots[op].store(
                s=states[op][0],
                r=-move_step,
                s_=states[op][1],
                done=True
            )
            env.render()
            print(env.last_move + 1)
            print(f'游戏结束，{bots_name[player]}获胜')
            break

        if move_step == 2:
            move_step = 0
            states
            player = (player + 1) % 2
            bots[player].store(
                s=states[player][0],
                r=0,
                s_=states[player][1],
                done=False
            )
            states[player][0] = states[(player + 1) % 2][1]
