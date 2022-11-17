"""
Use neural networks to get value and/or policy
Maybe add tree-search


state    p 1 2 3 4 5 6 # 1 2 3 4 5 6 #      (15)
output   1 2 3 4 5 6 v                      (7)
legal    1 2 3 4 5 6                        (6)
move     n                                  (int)

"""
import numpy as np
from numpy import ndarray
try:
    from game import Game
    from misc import choose
except ImportError:
    from .game import Game
    from .misc import choose


class Agent:
    """
    An agent takes as input the vector representation of the board and
    returns as output a distribution of possible moves
    By default, the agent is random

    Agent: state vector -> action vector

    """
    def __init__(self, function):
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
        s = input_state.get_state_representation()
        legal = input_state.get_legal_moves()
        return self.get_clean_function_output(s, legal)

    def __call__(self, input_state: Game) -> int:
        """
        Takes as input the state, returns legal move, choose or argmax on .compute_output()

        :param input_state:
        :return: move
        """
        v = self.compute_output(input_state)
        return choose(v)[0]


class RandomAgent(Agent):
    """ This bot plays random moves """

    def __init__(self):
        super().__init__(function=None)

    def __call__(self, input_state: Game) -> int:
        return choose(input_state.get_legal_moves())[0]


def fun(mat):
    out = np.array(list(range(1, 7))) * .1
    for i in range(6):
        out[i] += mat[6-i, i+1] * (i + 2)
    return out


class BetterAgent(Agent):
    """ Very simple hand-crafted bot that plays decently """
    def __call__(self, input_state: Game) -> int:
        v = self.compute_output(input_state)
        return int(np.argmax(v))
