# coding: utf-8
# author: Keavnn

import time
from pprint import pprint

from absl import app, flags

from game.connect6_mcts_rl import C6
from player.battle_players import BattleRandomPlayer
from player.mcts_rl import MCTS_POLICY, MCTSRL
from utils.config import load_config

flags.DEFINE_integer('board_size', 37, '棋盘尺寸大小')
flags.DEFINE_integer('box_size', 19, 'box的大小')
flags.DEFINE_string('model_path', './models', '指定要加载的模型文件夹')
flags.DEFINE_string('ip', '58.199.162.110', '指定服务器IP地址')
flags.DEFINE_string('port', '8080', '指定服务器端口号')


def battle_loop(env, player1, player2=None, is_black=True):
    if player2 is None:
        return
    env.reset()
    players = [player1, player2] if is_black else [player2, player1]

    while True:
        current = players[env.current_player]
        x, y = current.choose_action(env, return_prob=False, is_self_play=False)
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


def main(_argv):
    config = load_config('./battle_config.yaml')
    board_size = flags.FLAGS.board_size
    box_size = flags.FLAGS.box_size
    model_path = flags.FLAGS.model_path + '/models' + str(box_size)
    pprint(config)

    env = C6(dim=board_size, box_size=box_size)
    net = MCTS_POLICY(state_dim=[box_size, box_size, 4])
    net.restore(model_path)
    player = MCTSRL(
        pv_net=net,
        temp=config['temp'],
        c_puct=config['c_puct'],
        playout_num=config['playout_num'],
        dim=box_size,
        name='mcts_rl_policy',
    )
    player2 = BattleRandomPlayer()
    battle_loop(env, player, player2, is_black=True)


if __name__ == '__main__':
    try:
        app.run(main)
    except SystemExit:
        pass
