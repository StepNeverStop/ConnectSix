import sys

import numpy as np

from player.policy import MyPolicy
from utils.GymRender import GymRender


def _choose_valid_action(player, env_or_state, use_env=True):
    """Retry until the player returns a legal move."""
    while True:
        try:
            x, y = player.choose_action(env_or_state)
            if use_env:
                is_ok, _msg = env_or_state.can_place(x, y)
                if not is_ok:
                    continue
            return x, y
        except KeyboardInterrupt:
            print('游戏中断')
            sys.exit()


def _apply_turn(env, player, now_player, move_step):
    """Apply one move and return updated turn state."""
    x, y = _choose_valid_action(player, env, use_env=True)
    print(player.name, x, y)
    env.step(x, y)
    move_step += 1
    end, winner = env.is_over()
    if end:
        return move_step, now_player, True, winner
    if move_step == 2:
        return 0, (now_player + 1) % 2, False, None
    return move_step, now_player, False, None


def battle(env, players):
    """Headless battle loop for model evaluation."""
    env.register(players[0].name, players[-1].name)
    env.reset()
    now_player = 0
    move_step = 1

    while True:
        if now_player == 0:
            print('黑子')
        else:
            print('白子')
        move_step, now_player, finished, winner = _apply_turn(
            env, players[now_player], now_player, move_step
        )
        if finished:
            print(env.last_move)
            print(f'游戏结束，{players[now_player].name}获胜')
            break


def test_loop(env, players):
    render = GymRender()
    env.register(players[0].name, players[-1].name)
    env.reset()
    now_player = 0
    move_step = 1

    while True:
        env.render()
        if now_player == 0:
            print('黑子')
        else:
            print('白子')
        move_step, now_player, finished, winner = _apply_turn(
            env, players[now_player], now_player, move_step
        )
        if finished:
            env.render()
            render.render(env)
            print(env.last_move)
            print(f'游戏结束，{players[now_player].name}获胜')
            input()
            break


def train_loop(env, players):
    info = ['黑子', '白子']
    offset = [2, 1]
    wins = [0, 0]

    for episode in range(10000):
        if np.random.rand() > 0.5:
            players.reverse()
            wins.reverse()
        for i in range(2):
            if isinstance(players[i], MyPolicy):
                players[i].update_offset(offset[i])

        env.register(players[0].name, players[-1].name)

        state = env.reset()
        now_player = 0
        move_step = 1
        total_step = 0
        states = [
            [state, state],
            [state, state],
        ]

        while True:
            total_step += 1
            while True:
                try:
                    x, y = players[now_player].choose_action(state)
                    is_ok, _msg = env.can_place(x, y)
                    if not is_ok:
                        continue
                    break
                except KeyboardInterrupt:
                    print('游戏中断')
                    sys.exit()

            state = env.step(x, y)
            states[now_player][-1] = state
            move_step += 1
            if wins[now_player] - wins[(now_player + 1) % 2] < 6:
                players[now_player].learn()

            end, winner = env.is_over()
            if end:
                if winner == -1:
                    r0, r1 = 0.1, 0.1
                    print(f'episode: {episode}, step: {total_step:>3d}, 平局')
                else:
                    r0, r1 = 1, -move_step
                    print(
                        f'episode: {episode:>4d}, step: {total_step:>3d}, '
                        f'{info[now_player]}{players[now_player].name}获胜'
                    )
                    wins[now_player] += 1

                players[now_player].store(
                    s=states[now_player][0], r=r0,
                    s_=states[now_player][1], done=True,
                )
                players[now_player].writer_loop_summary(episode, reward=r0, step=total_step)
                next_player = (now_player + 1) % 2
                players[next_player].store(
                    s=states[next_player][0], r=r1,
                    s_=states[next_player][1], done=True,
                )
                players[next_player].writer_loop_summary(
                    episode, reward=r1, step=total_step
                )
                break

            if move_step == 2:
                move_step = 0
                now_player = (now_player + 1) % 2
                players[now_player].store(
                    s=states[now_player][0], r=0,
                    s_=states[now_player][1], done=False,
                )
                states[now_player][0] = states[(now_player + 1) % 2][1]
