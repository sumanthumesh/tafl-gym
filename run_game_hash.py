from gym_tafl.envs._game_engine import *
import neat


state_table = dict()
gen = 0


def eval_genomes(genomes, config):
    """
    runs the simulation of the current population of
    birds and sets their fitness based on the distance they
    reach in the game.
    """
    gen += 1

    # start by creating lists holding the genome itself, the
    # neural network associated with the genome and the
    # bird object that uses that network to play
    nets = []
    ge = []
    for genome_id, genome in genomes:
        genome.fitness = 0
        net = neat.nn.FeedForwardNetwork.create(genome, config)
        nets.append(net)
        ge.append(genome)

    return


def run_tournament():
    pass

def run_game(player1,player2):
    '''
    Function takes in two genomes and plays the game with these two
    For each new state reached, it updates a hash table with the state, the previous moves and current player
    Once the game ends, it increments a counter for each state to show if it is a win, loss or draw
    player1 is ATK and player2 is DEF
    '''
    #Create game engine
    game = GameEngine('gym_tafl/variants/custom.ini')
    #Intialize board
    board = np.zeros((game.n_rows,game.n_cols)) 
    game.fill_board(board)
    prev_moves = []
    #Run game till endgame or 150 moves
    for m in range(150):
        player = ATK if m%2==0 else DEF
        net1 = neat.nn.FeedForwardNetwork.create(player1,config)        
        net2 = neat.nn.FeedForwardNetwork.create(player2,config)      
        net = net1 if m%2==0 else net2  
        #Get all legal moves for current state
        moves = game.legal_moves(board,player)
        #For every move, find the next state and check its value from our genome
        best_move = moves[0]
        best_value = -100
        for move in moves:
            temp_move = decimal_to_space(move,game.n_rows,game.n_cols)
            temp_board = board.copy()
            game.apply_move(temp_board,temp_move)
            #Check the value that this new temp_board state gets from our genome
            temp_value = net.activate(temp_board)
            #Update best move and value
            if temp_value > best_value:
                best_value = temp_value
                best_move = move
        #Keep all the moves that are run on this game, we will need it to populate the hash table
        prev_moves.append(board)
        #Apply the best move on the board
        res = game.apply_move(board,best_move)
        if res.get('game_over') == True:
            break
    #Check endgame scenario
    end = game.check_endgame(prev_moves[-8:],best_move,player,m)
    #Set the increment values [atk_win, draw, def_win]
    if end.get('winner') == ATK:
        inc = [1,0,0]
    elif end.get('winner') == DEF:
        inc = [0,0,1]
    else:
        inc = [0,1,0]
    #Go through each and every state we encountered during the game and apply the correct increments
    new_board = np.zeros((game.n_rows,game.n_cols)) 
    game.fill_board(new_board)
    for idx,move in enumerate(prev_moves):
        player = ATK if idx%2==0 else DEF
        game.apply_move(new_board,decimal_to_space(move,game.n_rows,game.n_cols))
        #Create our flattened list that we will send as input to network
        net_inp = new_board.tolist() + prev_moves[idx:idx+8] + [player]
        #Update in the hash table
        if frozenset(net_inp) not in state_table.keys():
            state_table[frozenset(net_inp)] = inc
        else:
            state_table[frozenset(net_inp)][0] += inc[0]
            state_table[frozenset(net_inp)][1] += inc[1]
            state_table[frozenset(net_inp)][2] += inc[2]


def run(config_file):
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         config_file)
    p = neat.Population(config)

    # Report progress to console
    p.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    p.add_reporter(stats)
    p.add_reporter(neat.Checkpointer(5))

    winner = p.run(eval_genomes, 100) # arbitrarily picking 100 generations for now

    print('\nWinner winner chicken dinner:\n{!s}'.format(winner))
    
    """ maybe mess around with this given time

    # Shows output of the most fit genome against training data.
    print('\nOutput:')
    winner_net = neat.nn.FeedForwardNetwork.create(winner, config)
    for xi, xo in zip(xor_inputs, xor_outputs):
        output = winner_net.activate(xi)
        print("input {!r}, expected output {!r}, got {!r}".format(xi, xo, output))

    node_names = {-1: 'A', -2: 'B', 0: 'A XOR B'}
    visualize.draw_net(config, winner, True, node_names=node_names)
    visualize.draw_net(config, winner, True, node_names=node_names, prune_unused=True)
    visualize.plot_stats(stats, ylog=False, view=True)
    visualize.plot_species(stats, view=True)

    p = neat.Checkpointer.restore_checkpoint('neat-checkpoint-4')
    p.run(eval_genomes, 10)

    """

if __name__ == '__main__':
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, 'util/config.txt')
    genomes = []
    eval_genomes(genomes, config_path)