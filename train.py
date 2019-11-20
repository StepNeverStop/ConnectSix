# coding: utf-8
# athor: Keavnn
import os
import yaml
from pprint import pprint
from absl import app, flags, logging
from absl.flags import FLAGS
from game.connect6_mcts_rl import C6
from player.mcts_rl import MCTS_POLICY, MCTSRL

flags.DEFINE_integer('size', 19, '棋盘尺寸大小')
flags.DEFINE_float('learning_rate', 5e-4, '设置学习率')


def load_config(filename):
    if os.path.exists(filename):
        f = open(filename, 'r', encoding='utf-8')
    else:
        raise Exception('cannot find this config.')
    x = yaml.safe_load(f.read())
    f.close()
    return x


def main(_argv):
    config = load_config('./config.yaml')
    config['dim'] = FLAGS.size
    config['learning_rate'] = FLAGS.learning_rate
    pprint(config)
    env = C6(
        dim=config['dim']
    )
    net = MCTS_POLICY(
        state_dim=[config['dim'], config['dim'], 4],
        learning_rate=config['learning_rate'],
        buffer_size=config['buffer_size'],
        batch_size=config['batch_size']
    )
    player = MCTSRL(
        pv_net=net,
        temp=config['temp'],
        c_puct=config['c_puct'],
        playout_num=config['playout_num'],
        dim=config['dim'],
        name='mcts_rl_policy'
    )
    train_mcts_rl(env, player, config)


def train_mcts_rl(env, player, kwargs: dict):
    game_batch = kwargs.get('game_batch', 1600)
    game_batch_size = kwargs.get('game_batch_size', 1)
    for i in range(game_batch):
        for j in range(game_batch_size):
            env.self_play(player)
    pass


if __name__ == '__main__':
    try:
        app.run(main)
    except SystemExit:
        pass
