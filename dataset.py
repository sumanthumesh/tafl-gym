import os
import numpy as np
from gym_tafl.envs._game_engine import *
from gym_tafl.players import Player
import neat
import json

file_ptr = None

def write_to_file(winner,moves):
    file_ptr.writelines(f"{winner}: {moves}\n")

def run_game(turn_limit):
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
    #Run game till endgame or turn_limit moves
    for step in range(turn_limit):
        player = ATK if step%2==0 else DEF
        #Get all legal moves for current state
        moves = game.legal_moves(board,player)
        #If current player has no legal moves, then we stop game right now
        if len(moves) == 0:
            info['game_over'] = True
            info['winner'] = ATK if player == DEF else DEF
            info['reason'] = f"{player} Ran out of moves"
            break
        best_move = random.choice(moves)
        threefold = check_threefold_repetition_int(prev_moves,best_move,game.n_rows,game.n_cols)
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
            # print(f"Game Over")
            break
    # print(info)
    return (info,prev_moves)

def update_state_table(state_table, info, moves):
    game = GameEngine('gym_tafl/variants/custom.ini')
    board = np.zeros((game.n_rows,game.n_cols))
    game.fill_board(board)
    #Key for state table is the flattened board representation plus the player identifier
    for step,move in enumerate(moves):
        player = ATK if step%2==0 else DEF
        #Apply move
        game.apply_move(board,decimal_to_space(move,game.n_rows,game.n_cols))
        #Check if new state it is state table, if not add it and update counter
        inc = [0,0,0]
        if info['game_over'] == False:
            inc = [0,1,0]
        elif info['winner'] == DRAW:
            inc = [0,1,0]
        elif info['winner'] == ATK:
            inc = [1,0,0]
        elif info['winner'] == DEF:
            inc = [0,0,1]
        inp_to_hash = tuple(board.flatten().tolist() + [player])
        if inp_to_hash not in state_table.keys():
            state_table[inp_to_hash] = inc
        else:
            state_table[inp_to_hash][0] += inc[0]
            state_table[inp_to_hash][1] += inc[1]
            state_table[inp_to_hash][2] += inc[2]

def str_to_tuple(string:str) -> tuple[float]:
    #Remove end braces
    string = string[1:-1]
    #split it
    split_string = string.replace(" ","").split(",")
    l = []
    for s in split_string:
        l.append(float(s))
    return tuple(l)

def read_dataset(filepath):
    #Read filepath
    with open(filepath) as file:
        raw_dict = json.load(file)

    state_table = dict()
    for key,value in raw_dict.items():
        state_table[str_to_tuple(key)] = value
    
    return state_table

if __name__ == "__main__":
    file_ptr = open("dataset","w")
    atk_ctr = 0
    def_ctr = 0
    tie_ctr = 0
    state_table = dict()
    for i in range(100000):
        if i % 50 == 0:
            print(f"Game: {i}")
        #Run the game
        res = run_game(150)
        #Update the state table
        update_state_table(state_table,res[0],res[1])
        if res[0].get('game_over') == True:
            write_to_file(res[0]['winner'],res[1])
            if res[0]['winner'] == ATK:
                atk_ctr += 1
            elif res[0]['winner'] == DEF:
                def_ctr += 1
            else:
                tie_ctr += 1
        else:
            write_to_file(DRAW,res[1])
            tie_ctr += 1
    print(len(state_table))

    print(f"{atk_ctr}, {tie_ctr}, {def_ctr}")

    file_ptr.close()

    serializable_dict = {str(key): value for key, value in state_table.items()}

    with open("dataset.json","w") as file:
        json.dump(serializable_dict,file,indent=4)        

    # with open("dataset.json") as file:
    #     raw_dict = json.load(file)
    
    # print(type(list(raw_dict.values())[0]))

    # state_table = dict()
    # for key,value in raw_dict.items():
    #     state_table[str_to_tuple(key)] = value

    # print(state_table)    