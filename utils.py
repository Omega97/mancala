
def performance(agents, game, n_games):

    n_wins = 0

    for j in range(n_games):        
        p = j % 2
        result = game.play([agents[p], agents[1-p]])
        if p:
            n_wins += 1 - result
        else:
            n_wins += result

    winrate = (n_wins+1)/(n_games+2)
    std = (winrate * (1-winrate)/(n_games+2))**.5
    return winrate, std
