import numpy as np


def special_iter(player, index, n, size):
    """iterable used when making move on board """
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


def board_repr(p, go, s, board, result, space, n_round):
    prefix = f"{n_round//2+1:3})" + ' ' * 4
    side_1 = '  ' if p or go else '>>'
    side_1 += f" {list_to_str(board[0], space)}"
    t = max(s, 0)
    t = f"({t})" if t else '(.)'
    side_1 += f"  {t:>4}   "
    side_2 = '  ' if not p or go else '>>'
    side_2 += f" {list_to_str(board[1], space)}"
    t = max(-s, 0)
    t = f"({t})" if t else '(.)'
    side_2 += f"  {t:>4}   "
    suffix = ' '
    if go:
        suffix += 'w' if result else 'b'
        suffix += f" +{abs(s)}"
    return prefix + side_1 + side_2 + suffix


def list_to_str(v, space=2):
    return ' '.join([f'{i if i else ".":>{space}}' for i in v])


def abstract_vector_state(player, points, board, max_size):
    """ represent game state as list of numbers
    - player (1)
    - player board (board_size)
    - player points (1)
    - opponent board (board_size)
    - opponent points (1)
    """
    w = [points, -points]
    v = np.array([player], dtype=int)
    v = np.append(v, board[player])
    v = np.append(v, w[player])
    v = np.append(v, board[1-player])
    v = np.append(v, w[1-player])
    v = v.clip(0, max_size)
    return v


def choose(v) -> int:
    """
    choose random array index given probability distribution
    :param v: array
    :return:
    """
    s = sum(v)
    if not s:
        raise ValueError('Sum cannot be 0')
    return np.random.choice(len(v), 1, p=v/s)[0]


def argmax(v) -> int:
    return int(np.argmax(v))
