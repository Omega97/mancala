import numpy as np


def special_iter(player, index, n, size):
    """"""
    assert player in (0, 1)
    t_skip = (True, player, index)
    on_board = True
    for _ in range(n):
        if not on_board:
            on_board = True
            player = 1 - player
        else:
            index += 1
            if index >= size:
                index = 0
                on_board = False
        out = (on_board, player, index)
        if out != t_skip:
            yield out


def list_to_str(v, space=2):
    return ' '.join([f'{i if i else ".":>{space}}' for i in v])


def build_binary_matrix(v, max_size):
    """convert array to one-hot binary matrix of limited size"""
    mat = np.zeros((max_size+1, len(v)), dtype=int)
    for i in range(len(v)):
        mat[v[i], i] = 1
    return mat


def abstract_state(player, points, board, max_size):
    w = [points, 1-points]
    v = np.array([player], dtype=int)
    v = np.append(v, board[player])
    v = np.append(v, w[player])
    v = np.append(v, board[1-player])
    v = np.append(v, w[1-player])
    v = v.clip(0, max_size)
    return build_binary_matrix(v, max_size)
