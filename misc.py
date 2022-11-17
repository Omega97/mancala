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


def build_binary_vector(n, max_size):
    """convert array to one-hot binary matrix of limited size"""
    v = np.zeros(max_size+1, dtype=int)
    v[n] = 1
    return v


def build_binary_matrix(v, max_size):
    """convert array to one-hot binary matrix of limited size"""
    mat = np.zeros((max_size+1, len(v)), dtype=int)
    for i in range(len(v)):
        mat[v[i], i] = 1
    return mat


def abstract_vector_state(player, points, board, max_size):
    w = [points, 1-points]
    v = np.array([player], dtype=int)
    v = np.append(v, board[player])
    v = np.append(v, w[player])
    v = np.append(v, board[1-player])
    v = np.append(v, w[1-player])
    v = v.clip(0, max_size)
    return v


def abstract_binary_matrix_state(player, points, board, max_size):
    v = abstract_vector_state(player, points, board, max_size)
    return build_binary_matrix(v, max_size)


def abstract_binary_vector_state(player, points, board, max_size):
    """all information of state encoded in binary vector"""
    v = abstract_vector_state(player, points, board, max_size)
    out = build_binary_vector(v[0], 1)
    for i in v[1:]:
        w = build_binary_vector(i, max_size)
        out = np.append(out, w)
    return out


def choose(v, n=1) -> np.ndarray:
    """
    choose random array index given probability distribution
    :param v: array
    :param n: number of output samples
    :return:
    """
    s = sum(v)
    if not s:
        raise ValueError('Sum cannot be 0')
    return np.random.choice(len(v), n, p=v/s)
