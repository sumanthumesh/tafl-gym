from gym_tafl.envs._game_engine import *

import neat

state_table = dict()

def eval_genomes(genomes,config):
    print("Reached here 1")
    #Pick the first two genomes
    g1 = genomes[0][1]
    g2 = genomes[1][1]
    #Pass these two genomes as input to run game
    run_game(g1,g2,config)
    print("Reached here 2")

def pad_prev_moves(prev_moves,size):
    '''
    Pad the prev moves list with correct number of 0s where needed
    '''
    if len(prev_moves) < size:
        padded_moves = [0]*(size-len(prev_moves)) + prev_moves
    else:
        padded_moves = prev_moves[-8:]
    return padded_moves

def run_game(player1,player2,config):
    '''
    Function takes in two genomes and plays the game with these two
    For each new state reached, it updates a hash table with the state, the previous moves and current player
    Once the game ends, it increments a counter for each state to show if it is a win, loss or draw
    player1 is ATK and player2 is DEF
    '''
    info ={
        'game_over' : False,
        'winner' : None,
        'reason' : None 
    }
    #Create game engine
    game = GameEngine('gym_tafl/variants/custom.ini')
    #Intialize board
    board = np.zeros((game.n_rows,game.n_cols)) 
    game.fill_board(board)
    prev_moves = []
    #Run game till endgame or 150 moves
    num_steps=150
    for step in range(num_steps):
        player = ATK if step%2==0 else DEF
        net1 = neat.nn.FeedForwardNetwork.create(player1,config)        
        net2 = neat.nn.FeedForwardNetwork.create(player2,config)      
        net = net1 if step%2==0 else net2  
        #Get all legal moves for current state
        moves = game.legal_moves(board,player)
        #If current player has no legal moves, then we stop game right now
        if len(moves) == 0:
            info['game_over'] = True
            info['winner'] = ATK if player == DEF else DEF
            info['reason'] = f"{player} Ran out of moves"
            break
        #For every move, find the next state and check its value from our genome
        best_move = moves[0]
        best_value = [-100]
        for idx,move in enumerate(moves):
            temp_move = decimal_to_space(move,game.n_rows,game.n_cols)
            temp_board = board.copy()
            game.apply_move(temp_board,temp_move)
            #Check the value that this new temp_board state gets from our genome
            padded_moves = pad_prev_moves(prev_moves,8)
            net_inp = temp_board.flatten().tolist() + padded_moves + [player]
            # print(f"{step} {idx}")
            # print(f"{temp_board}")
            # print(f"{padded_moves}")
            # print(f"{player}\n")
            temp_value = net.activate(net_inp)
            #Update best move and value
            if temp_value[0] > best_value[0]:
                best_value = temp_value
                best_move = move
        #Check if we have 3 fold repitition
        threefold = check_threefold_repetition_int(pad_prev_moves(prev_moves,8),best_move,game.n_rows,game.n_cols)
        if threefold:
            info['game_over'] = True
            info['winner'] = DRAW
            info['reason'] = 'Threefold repitition'
        #Keep all the moves that are run on this game, we will need it to populate the hash table
        prev_moves.append(best_move)
        #Apply the best move on the board
        res = game.apply_move(board,decimal_to_space(best_move,game.n_rows,game.n_cols))
        if res.get('game_over') == True:
            info['game_over'] = True
            info['winner'] = player
            info['reason'] = f"{player} Won"
            num_steps = step
            print(f"Game Over")
            break
    #Check endgame scenario
    #Consolidate the results
    if info.get('game_over') == False:
        inc = [0,0,0]
    elif info.get('winner') == ATK:
        inc = [1,0,0]
    elif info.get('winner') == DEF:
        inc = [0,0,1]
    elif info.get('winner') == DRAW:
        inc = [0,1,0]
    #Go through each and every state we encountered during the game and apply the correct increments
    new_board = np.zeros((game.n_rows,game.n_cols)) 
    game.fill_board(new_board)
    for idx,move in enumerate(prev_moves):
        player = ATK if idx%2==0 else DEF
        game.apply_move(new_board,decimal_to_space(move,game.n_rows,game.n_cols))
        #Create our flattened list that we will send as input to network
        net_inp = new_board.flatten().tolist() + prev_moves[idx:idx+8] + [player]
        #Update in the hash table
        tup_net_inp = tuple(net_inp)
        if tup_net_inp not in state_table.keys():
            state_table[tuple(net_inp)] = inc
        else:
            state_table[tuple(net_inp)][0] += inc[0]
            state_table[tuple(net_inp)][1] += inc[1]
            state_table[tuple(net_inp)][2] += inc[2]
    print(info)

if __name__ == '__main__':
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         'neat.config')
    p = neat.Population(config)
    p.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    p.add_reporter(stats)
    p.add_reporter(neat.Checkpointer(5))

    winner = p.run(eval_genomes,2)