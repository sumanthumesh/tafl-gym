import os
import numpy as np
from gym_tafl.envs._game_engine import *
from gym_tafl.players import Player
import neat

class HashTournament():
    """
    May the best viking chess bot win
    """

    def __init__(self, game, turn_limit=150, game_scores={'win': 1, 'draw':0, 'loss':-1}) -> None:
        self.game = game
        self.turn_limit = turn_limit
        self.game_scores = game_scores
        self.state_table = {}

    def run_tournament(self, players):
        # round robin tourney
        for idxA, playerA in players:
            for idxB, playerB in players[idxA + 1:]:
                #don't let players play against themselves
                if idxA == idxB:
                    continue
                #play 2 games so each player can play both attacker and defender
                # print(f"Game: Player {idxA} VS {idxB}")
                res = self.run_game(playerA, playerB)
                # print(res)
                # self.update_scores(res, idxA, idxB)

    def run_game(self, player1, player2):
        '''
        Function takes in two genomes and plays the game with these two
        For each new state reached, it updates a hash table with the state, the previous moves and current player
        Once the game ends, it increments a counter for each state to show if it is a win, loss or draw
        player1 is ATK and player2 is DEF
        '''

        #Intialize board
        board = np.zeros((self.game.n_rows, self.game.n_cols)) 
        self.game.fill_board(board)
        prev_moves = []

        #Run game till endgame or 150 moves
        num_steps=150
        for step in range(1000):
            player = ATK if step % 2 == 0 else DEF
            net = player1.net if step % 2 == 0 else player2.net

            #Get all legal moves for current state
            moves = self.game.legal_moves(board, player)
            
            # For every move, find the next state and check its value from our genome
            temp_player = Player(1, net, self.game, step % 2 == 0)
            best_move = temp_player.choose_move(board, moves)

            """
            #For every move, find the next state and check its value from our genome
            best_move = moves[0]
            best_value = [-100]
            for idx, move in enumerate(moves):
                temp_move = decimal_to_space(move, self.game.n_rows, self.game.n_cols)
                temp_board = board.copy()
                self.game.apply_move(temp_board,temp_move)
                #Check the value that this new temp_board state gets from our genome
                if len(prev_moves) < 8:
                    padded_moves = [0]*(8-len(prev_moves)) + prev_moves
                else:
                    padded_moves = prev_moves[-8:]
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
            """
            
            #Keep all the moves that are run on this game, we will need it to populate the hash table
            prev_moves.append(best_move)

            #Apply the best move on the board
            best_move_tuple = decimal_to_space(best_move, self.game.n_rows, self.game.n_cols)
            res = self.game.apply_move(board, best_move_tuple)
            if res.get('game_over') == True:
                num_steps = step
                break

        #Check endgame scenario
        end = self.game.check_endgame(prev_moves[-8:], best_move_tuple, player, step)
        if end['winner'] != -1:
            print(end)
        #Set the increment values [atk_win, draw, def_win]
        if end.get('winner') == ATK:
            inc = [1,0,0]
        elif end.get('winner') == DEF:
            inc = [0,0,1]
        elif end.get('winner') == DRAW:
            inc = [0,1,0]
        else:
            inc = [0,0,0]

        print(f"Game ended after {num_steps} steps")
        print(f"{inc}")

        #Go through each and every state we encountered during the game and apply the correct increments
        new_board = np.zeros((self.game.n_rows, self.game.n_cols)) 
        self.game.fill_board(new_board)
        for idx,move in enumerate(prev_moves):
            player = ATK if idx%2==0 else DEF
            self.game.apply_move(new_board,decimal_to_space(move, self.game.n_rows, self.game.n_cols))
            #Create our flattened list that we will send as input to network
            net_inp = new_board.flatten().tolist() + prev_moves[idx:idx+8] + [player]

            #Update in the hash table
            tup_net_inp = tuple(net_inp)
            if tup_net_inp not in self.state_table.keys():
                self.state_table[tuple(net_inp)] = inc
            else:
                self.state_table[tuple(net_inp)][0] += inc[0]
                self.state_table[tuple(net_inp)][1] += inc[1]
                self.state_table[tuple(net_inp)][2] += inc[2]
        # print(state_table)


def eval_genomes(genomes, config):
    game = GameEngine('gym_tafl/variants/custom.ini')

    players = []
    for id, genome in genomes:
        genome.fitness = 0
        players.append((id, Player(0.1, neat.nn.FeedForwardNetwork.create(genome, config), game, -1)))
        
    tournament = HashTournament(game)
    tournament_state = tournament.run_tournament(players)

    # for id, genome in genomes:
    #     genome.fitness = [something]
    
    return


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

    p = neat.Checkpointer.restore_checkpoint('neat-checkpoint-4')
    p.run(eval_genomes, 10)

    """

if __name__ == '__main__':
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, 'config-feedforward.txt')
    run(config_path)