# coding: utf-8
# author: Keavnn

import time

from absl import app, flags

from game import Connect6WJS
from loop import battle, test_loop, train_loop

flags.DEFINE_integer('size', 19, '棋盘尺寸大小')
flags.DEFINE_enum('mode', 'test', ['train', 'test', 'battle'],
                  'train: 训练模型, '
                  'test: 测试, '
                  'battle: 对战模型，我方模型控制一子，不需要render')
flags.DEFINE_enum('p1_mode', 'random', ['random', 'player', 'mcts'],
                  'random: 随机策略, '
                  'player: 手动控制, '
                  'mcts: 使用蒙特卡洛树搜索模型')
flags.DEFINE_string('p1_name', 'player1', '设置第一位选手的名字')
flags.DEFINE_enum('p2_mode', 'random', ['random', 'player', 'mcts'],
                  'random: 随机策略, '
                  'player: 手动控制, '
                  'mcts: 使用蒙特卡洛树搜索模型')
flags.DEFINE_string('p2_name', 'player2', '设置第二位选手的名字')
flags.DEFINE_float('learning_rate', 5e-4, '设置学习率')


def generate_model(choice, param):
    if choice == 'player':
        from player import Player
        return Player(**param)
    if choice == 'random':
        from player import RandomBot
        return RandomBot(**param)
    if choice == 'mcts':
        from player import MCTSPlayer
        return MCTSPlayer(
            name=param['name'], c_puct=5, n_playout=1000, max_step=1000
        )
    from player import MyPolicy
    return MyPolicy(**param)


def main(_argv):
    dim = flags.FLAGS.size
    player1_param = {'dim': dim, 'name': flags.FLAGS.p1_name}
    player2_param = {'dim': dim, 'name': flags.FLAGS.p2_name}

    model1 = generate_model(flags.FLAGS.p1_mode, player1_param)
    time.sleep(1)
    model2 = generate_model(flags.FLAGS.p2_mode, player2_param)

    players = [model1, model2]
    env = Connect6WJS(dim=dim)

    if flags.FLAGS.mode == 'test':
        test_loop(env, players)
    elif flags.FLAGS.mode == 'train':
        train_loop(env, players)
    elif flags.FLAGS.mode == 'battle':
        battle(env, players)
    else:
        raise NotImplementedError


if __name__ == '__main__':
    try:
        app.run(main)
    except SystemExit:
        pass
