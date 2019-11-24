# coding: utf-8
# athor: Keavnn
import numpy as np
from pprint import pprint
from absl import app, flags, logging
from absl.flags import FLAGS
from game.connect6_mcts_rl import C6
from player.mcts_rl import MCTS_POLICY, MCTSRL
from utils.timer import timer
from utils.sth import load_config

flags.DEFINE_integer('size', 19, '棋盘尺寸大小')
flags.DEFINE_boolean('load', False, '是否载入模型')
flags.DEFINE_float('learning_rate', 5e-4, '设置学习率')


def main(_argv):
    config = load_config('./train_config.yaml')
    config['dim'] = FLAGS.size
    config['learning_rate'] = FLAGS.learning_rate
    cp_dir = './models' + str(FLAGS.size)
    pprint(config)
    env = C6(
        dim=config['dim']
    )
    net = MCTS_POLICY(
        state_dim=[config['dim'], config['dim'], 4],
        learning_rate=config['learning_rate'],
        buffer_size=config['buffer_size'],
        batch_size=config['batch_size'],
        epochs=config['epochs'],
        cp_dir=cp_dir
    )
    if FLAGS.load:
        net.restore(cp_dir=cp_dir)
    player = MCTSRL(
        pv_net=net,
        temp=config['temp'],
        c_puct=config['c_puct'],
        playout_num=config['playout_num'],
        dim=config['dim'],
        name='mcts_rl_policy'
    )
    eval_net = MCTS_POLICY(
        state_dim=[config['dim'], config['dim'], 4],
        learning_rate=config['learning_rate'],
        buffer_size=config['buffer_size'],
        batch_size=config['batch_size'],
        epochs=config['epochs']
    )
    eval_player = MCTSRL(
        pv_net=net,
        temp=config['temp'],
        c_puct=config['c_puct'],
        playout_num=config['playout_num'],
        dim=config['dim'],
        name='eval_policy'
    )
    train_mcts_rl(env, player, eval_player, config)


def augment_data(dim, data):
    '''
    增广数据集
    '''
    extend_data = []
    for state, mcts_porb, winner in data:
        for i in [1, 2, 3, 4]:
            # rotate counterclockwise
            equi_state = np.array([np.rot90(s, i) for s in state])
            equi_mcts_prob = np.rot90(np.flipud(
                mcts_porb.reshape(dim, dim)), i)
            extend_data.append((equi_state,
                                np.flipud(equi_mcts_prob).flatten(),
                                winner))
            # flip horizontally
            equi_state = np.array([np.fliplr(s) for s in equi_state])
            equi_mcts_prob = np.fliplr(equi_mcts_prob)
            extend_data.append((equi_state,
                                np.flipud(equi_mcts_prob).flatten(),
                                winner))
    return extend_data


@timer
def evaluate(num, ratio, env, player1, player2):
    '''
    参数：
        num: 评估对弈的盘数
        ratio: 用于判断是否有进步的取胜比率
        env: 游戏环境
        player1: 第一位选手
        player2: 第二位选手
    '''
    win_count = 0
    for i in range(num):
        env.reset()
        flag = env.play(player1, player2)
        if flag:
            win_count += 1
        logging.info(f'本轮第{i}次评估对局，结果为{flag}')
    eval_ratio = win_count / num
    logging.info(f'本轮测试评估的胜率为{eval_ratio}')
    if eval_ratio > ratio:
        return True
    else:
        return False


def train_mcts_rl(env, player, eval_player, kwargs: dict):
    game_batch = kwargs.get('game_batch', 1600)
    game_batch_size = kwargs.get('game_batch_size', 1)
    save_frequent = kwargs.get('save_frequent', 10)
    eval_num = kwargs.get('eval_num', 100)
    ratio = kwargs.get('ratio', 0.55)
    eval_interval = kwargs.get('eval_interval', 20)
    player.net.save_checkpoint(0)

    for i in range(game_batch):
        logging.info(f'-> 第{i}批次训练')
        for j in range(game_batch_size):
            logging.info(f'--> 第{i}批次第{j}次训练')
            data = env.self_play(player)
            data = list(data)[:]
            data = augment_data(env.dim, data)
            player.net.store(data)
        player.net.learn()
        logging.info(f'模型第{i}批次已学习')

        if i % eval_interval == 0 and i != 0:
            eval_player.net.restore(cp_dir='./models')
            ret = evaluate(eval_num, ratio, env, player, eval_player)
            if ret:
                player.net.save_checkpoint(i)
                logging.info(f'评估结束, 优化模型已保存')
            continue

        if i % save_frequent == 0:
            player.net.save_checkpoint(i)
            logging.info(f'第{i}次保存模型')


if __name__ == '__main__':
    try:
        app.run(main)
    except SystemExit:
        pass
