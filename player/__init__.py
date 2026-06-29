from .random_bot import RandomBot
from .player import Player
from .policy import MyPolicy
from .mcts import MCTSPlayer
from .mcts_rl import MCTS_POLICY, MCTSRL
from .battle_players import BattleRandomPlayer, BattleHumanPlayer, RemotePlayer

# Backward-compatible aliases
RandomPlayer = BattleRandomPlayer
HumanPlayer = BattleHumanPlayer
TestPlayer = RemotePlayer
