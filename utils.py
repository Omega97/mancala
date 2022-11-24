try:
    from game import Game
except ImportError:
    from game import Game


def performance(agents, game: Game, n_games):
    """
    agents play against each other, calculates expected win-rate of first agent going first
    :param agents: list of two agents, they will play each other
    :param game: starting position
    :param n_games:
    :return:
    """
    n_wins = 0
    for j in range(n_games):
        n_wins += game.play(agents)
    win_rate = (n_wins+1)/(n_games+2)
    std = (win_rate * (1-win_rate)/(n_games+2))**.5
    return win_rate, std
