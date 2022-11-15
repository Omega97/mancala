"""
Use neural networks to get value and/or policy
Maybe add tree-search


state    p 1 2 3 4 5 6 # 1 2 3 4 5 6 #      (15)
output   1 2 3 4 5 6 v                      (7)
legal    1 2 3 4 5 6 1                      (7)
move     n                                  (int)

"""
from numpy import ndarray


class Agent:
    """
    An agent takes as input the vector-representation of the board and
    returns as output a distribution of possible moves

    Agent: state vector -> action vector

    """
    def __init__(self, function):
        """

        :param function: neural network, callable
        """
        self.function = function

    def get_clean_function_output(self, input_state: ndarray, legal_moves: ndarray) -> ndarray:
        """
        Takes as input the abstract representation of the state (binary matrix)
        :param input_state: abstract representation of input state
        :param legal_moves: legal moves array
        :return: move distribution given by the function + value, (illegal moves are set to 0)
        """
        out = self.function(input_state)
        if out.shape != legal_moves.shape:
            raise IndexError(f'Agent did not return output of correct shape '
                             f'({out.shape} != {legal_moves.shape})')
        return out * legal_moves

    def __call__(self, input_state: ndarray, legal_moves: ndarray) -> int:
        """
        Takes board state as input
        returns: move
        """
        raise NotImplemented
