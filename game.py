"""
The Game object can be used to play a game of Mancala (custom rules)

starting position, white's play, black starts with 6 points of handicap for going second
  1)    >>  4  4  4  4  4  4  (.)       4  4  4  4  4  4  (6)

middle game, both players have even points
  6)    >>  2  1  5  2  .  1  (.)       3 10  1  .  .  1  (.)

game over, black won by +2 points
 12)        .  .  .  .  .  .  (.)       .  1  1  .  .  .  (2)    b +2

"""
import numpy as np
try:
    from misc import abstract_binary_vector_state, list_to_str, special_iter
except ImportError:
    from .misc import abstract_binary_vector_state, list_to_str, special_iter


class Game:
    """
    This class describes the game
    State: action vector -> new state
    white = first
    black = second
    To win: white needs to have more points than black to win
    """
    board_size = 6
    starting_stones = 4
    handicap = 6    # black starts the game with this many points
    max_size = 18   # numbers higher than this are not distinguished

    def __init__(self):
        self.state = None
        self.history = []   # lit of (player, state_repr, move)
        self.init_board()

    def __repr__(self, space=2):
        if self.state is None:
            return 'Game()'
        p = self.get_player()
        go = self.is_game_over()
        s = self.get_points()

        prefix = f"{self.state['round']//2+1:3})" + ' ' * 4

        side_1 = '  ' if p or go else '>>'
        side_1 += f" {list_to_str(self.state['board'][0], space)}"
        t = max(s, 0)
        t = f"({t})" if t else '(.)'
        side_1 += f"  {t:>4}   "

        side_2 = '  ' if not p or go else '>>'
        side_2 += f" {list_to_str(self.state['board'][1], space)}"
        t = max(-s, 0)
        t = f"({t})" if t else '(.)'
        side_2 += f"  {t:>4}   "

        suffix = ' '
        if go:
            suffix += 'w' if self.get_game_result() else 'b'
            suffix += f" +{abs(s)}"

        return prefix + side_1 + side_2 + suffix

    def init_board(self):
        """setup the initial board state"""
        board = np.ones((2, Game.board_size), dtype=int) * Game.starting_stones
        self.history.clear()
        self.state = {'board': board,
                      'barns': np.zeros(2, dtype=int),
                      'player': 0,
                      'round': 0,
                      'ply': 0}

    def get_player(self):
        return self.state['player']

    def get_barns(self):
        return self.state['barns']

    def get_board(self):
        return self.state['board']

    def get_points(self):
        """white needs to exceed the black points by more than the handicap"""
        barn_1, barn_2 = self.get_barns()
        return barn_1 - barn_2 - Game.handicap

    def get_state_representation(self) -> np.ndarray:
        """
        return an array-type object that describes the board state from the player's point of view
        abstract state gets passed to agent
        features:
        - current player
        - pieces etc...
        """
        return abstract_binary_vector_state(self.get_player(), self.get_points(), self.get_board(), Game.max_size)

    def get_legal_moves(self) -> np.ndarray:
        """
        return an array-type object describing all the possible legal moves
        There must be at least one legal move ()
        """
        p = self.get_player()
        v = self.get_board()[p]
        return v.clip(0, 1)

    def is_game_over(self):
        p = self.get_player()
        board = self.get_board()[p]
        return sum(board) == 0

    def get_game_result(self):
        """
        1 = white won
        0 = black won
        None if not game-over
        """
        if self.is_game_over():
            return int(self.get_points() > 0)

    def get_history(self, player):
        return [(v, m) for p, v, m in self.history if p == player]

    def make_move(self, move: int):
        if self.state is None:
            self.init_board()
        move %= self.board_size
        p = self.get_player()
        n = self.get_board()[p][move]
        if not n:
            raise ValueError(f'Illegal move ({move})')
        self.state['board'][p][move] = 0
        on_board = False
        side = 0
        for on_board, side, index in special_iter(p, move, n, Game.board_size):
            if on_board:
                self.state['board'][side][index] += 1
            else:
                self.state['barns'][side] += 1

        self.state['ply'] += 1
        if on_board or side != p:
            self.state['player'] = 1 - p
            self.state['round'] += 1

    def play(self, agents, show=False, history=False):
        """perform a game start to finish, return outcome
        :param agents: list of agents; an agent takes as input
        :param show: print game to console
        :param history: if True, moves are saved along the board states and players
        """
        self.init_board()
        if show:
            print(self)
        while not self.is_game_over():
            p = self.state['player']
            move = agents[p](self)
            if history:
                self.history += [(p, self.get_state_representation(), move)]
            self.make_move(move)
            if show:
                print(self)
        return self.get_game_result()
