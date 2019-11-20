# coding: utf-8
# athor: Keavnn

import time
from absl import app, flags, logging
from absl.flags import FLAGS
from game import Connect6, Connect6WJS
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


def main(_argv):
    # 设置棋盘维度
    dim = FLAGS.size
    player1_name = FLAGS.p1_name
    player2_name = FLAGS.p2_name
    player1_param = dict([
        ['dim', dim],
        ['name', player1_name],
    ])
    player2_param = dict([
        ['dim', dim],
        ['name', player2_name],
    ])
    model1 = generate_model(FLAGS.p1_mode, player1_param)
    time.sleep(1)  # 避免log写在同一个文件
    model2 = generate_model(FLAGS.p2_mode, player2_param)

    players = [model1, model2]
    env = Connect6WJS(dim=dim)
    if FLAGS.mode == 'test':
        test_loop(env, players)
    elif FLAGS.mode == 'train':
        train_loop(env, players)
    elif FLAGS.mode == 'battle':
        battle(env, players)
    else:
        raise NotImplementedError


def generate_model(choice, param):
    if choice == 'player':
        from player import Player
        return Player(**param)
    elif choice == 'random':
        from player import RandomBot
        return RandomBot(**param)
    elif choice == 'mcts':
        from player import MCTSPlayer
        return MCTSPlayer(name=param['name'], c_puct=5, n_playout=1000, max_step=1000)
    else:
        from player import MyPolicy
        return MyPolicy(**param)


if __name__ == '__main__':
    try:
        app.run(main)
    except SystemExit:
        pass
