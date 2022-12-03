"""
The Agent is the object that plays the game.
Build an agent using a neural network and train it on games of self.play to perform self-RL.

state    p 1 2 3 4 5 6 # 1 2 3 4 5 6 #      (15)
output   1 2 3 4 5 6 v                      (7)
legal    1 2 3 4 5 6                        (6)
move     n                                  (int)

"""
import numpy as np
from numpy import ndarray
try:
    from game import Game
    from misc import choose, argmax
except ImportError:
    from .game import Game
    from .misc import choose, argmax


class Agent:
    """
    An agent takes as input the vector representation of the board and
    returns as output a distribution of possible moves
    By default, the agent is random

    Agent: state vector -> action vector

    """
    def __init__(self, function=None):
        """

        :param function: neural network, callable,
        """
        self.function = function

    def get_clean_function_output(self, state_repr: ndarray, legal_moves: ndarray) -> ndarray:
        """
        Takes as input the state, returns policy with only legal moves

        :param state_repr: abstract representation of input state (binary matrix)
        :param legal_moves: legal moves array
        :return: move distribution given by the function + value, (illegal moves are set to 0)
        """
        return self.function(state_repr) * legal_moves

    def compute_output(self, input_state: Game) -> ndarray:
        """
        Use self.function to compute move distribution (policy-like)
        If you want to do a tree-search, do it here

        :param input_state: used to chose move, possibly by using tree-search
        input state must have:
         * input_state.get_legal_moves()
         * input_state.get_state_representation()
        :return:
        """
        s = input_state.abstract_representation()
        legal = input_state.get_legal_moves()
        return self.get_clean_function_output(s, legal)

    def __call__(self, input_state: Game) -> int:
        """
        Pick move
        Takes as input the state, returns legal move, choose or argmax on .compute_output()

        :param input_state:
        :return: move
        """
        v = self.compute_output(input_state)
        return choose(v)


class RandomAgent(Agent):
    """ This bot plays random moves """
    def compute_output(self, input_state):
        return input_state.get_legal_moves()


class SimpleAgent(Agent):
    """
    Very simple hand-crafted bot that plays decently
    warning: deterministic!
    looks for exact matches, then prioritizes the rightmost squares
    """
    def __init__(self, max_size=Game.max_size, board_size=Game.board_size):
        super().__init__()
        self.max_size = max_size
        self.board_size = board_size
        self.biases = None
        self.weights = None
        self._set_parameters()

    def _set_parameters(self):
        input_size = 4 + 3 * self.board_size + 2 * (self.max_size + self.max_size * self.board_size)
        self.biases = np.array(list(range(1, self.board_size + 1)), dtype=float)
        self.weights = np.zeros((6, input_size), dtype=float)
        n = self.board_size + 1

        for i in range(self.board_size):
            j = (2 * n - 1) * self.board_size - 2 * n * (i-1) + i
            self.weights[i, j] += self.board_size + i + 1

    def get_clean_function_output(self, state_repr: ndarray, legal_moves: ndarray) -> ndarray:
        return legal_moves * (self.biases + self.weights.dot(state_repr))

    def __call__(self, input_state: Game) -> int:
        return argmax(self.compute_output(input_state))


class SimpleNoisyAgent(SimpleAgent):
    """
    Very simple hand-crafted bot that plays decently
    First moves are random
    looks for exact matches, then prioritizes the rightmost squares
    """
    def __init__(self, max_size, board_size, n_random_ply=0):
        super().__init__(max_size, board_size)
        self.n_random_ply = n_random_ply

    def __call__(self, input_state: Game) -> int:
        if input_state.state['ply'] < self.n_random_ply:
            return choose(input_state.get_legal_moves())
        v = self.compute_output(input_state)
        return argmax(v)
