import os
import numpy as np
from gym_tafl.envs._game_engine import *
from gym_tafl.players import Player
import neat
from pickle import dump
from math import ceil

class PureTournament():
    """
    Class that handles running tournaments between genome players
    """
    def __init__(self, game, turn_limit=150, game_scores={'win': 3, 'draw':1, 'loss':0}) -> None:
        self.game = game
        self.turn_limit = turn_limit
        self.game_scores = game_scores
        self.player_scores = []
        self.attacker_win_count = 0
        self.defender_win_count = 0
        self.draw_count = 0


    def play_tournament(self, players) -> list[int]:
        """
        Given list of genomes, run a round robin tournament and update each genome's fitness to be its score in the tournament
        """
        #create players from list of genomes. Each player starts as neither attacker nor defender
        # self.players = [Player(epsilon, neat.nn.FeedForwardNetwork.create(genome, config), self.game, -1) for _, genome in genomes]
        self.player_scores = [0 for _ in players]
        self.attacker_win_count = 0
        self.defender_win_count = 0
        self.draw_count = 0

        #run the round robin tournament
        for idxA, playerA in players:
            for idxB, playerB in players[idxA + 1:]:
                #don't let players play against themselves
                if idxA == idxB:
                    continue
                #play 2 games so each player can play both attacker and defender
                # print(f"Game: Player {idxA} VS {idxB}")
                res = self.play_game(playerA, playerB)
                # print(res)
                self.update_scores(res, idxA, idxB)
                # print(f"Game: Player {idxB} VS {idxA}")
                res = self.play_game(playerB, playerA)
                # print(res)
                self.update_scores(res, idxB, idxA)
                
        tournament_stats = {
            'scores': self.player_scores,
            'attacker_win_count': self.attacker_win_count,
            'defender_win_count': self.defender_win_count,
            'draw_count': self.draw_count
        }
        return tournament_stats
        
    def update_scores(self, res, idx1, idx2):
        if res['winner'] == ATK:
            self.attacker_win_count += 1
            self.player_scores[idx1] += self.game_scores['win']
            self.player_scores[idx2] += self.game_scores['loss']
        elif res['winner'] == DEF:
            self.defender_win_count += 1
            self.player_scores[idx1] += self.game_scores['loss']
            self.player_scores[idx2] += self.game_scores['win']
        else:
            self.draw_count += 1
            self.player_scores[idx1] += self.game_scores['draw']
            self.player_scores[idx2] += self.game_scores['draw']

    def play_game(self, player1, player2) -> dict:
        """
        Play game between player1 and player2. Player1 is the attacker and player2 is the defender.

        Return
        ------
        dict with 2 values:
        
        'winner': 1 if player1/attacker wins, 0 if player2/defender wins, or -1 if game ends in draw |
         
        'turns_played': number of turns it took for game to end
        """
        player1.set_role(ATK)
        player2.set_role(DEF)
        board = np.zeros((self.game.n_rows,self.game.n_cols)) 
        self.game.fill_board(board)

        last_moves = []

        for turn in range(self.turn_limit):
            player = player1 if turn % 2 == 0 else player2

            move = player.choose_move(board, last_moves)

            #make the move and return winner if move wins game
            result = self.game.alt_apply_move(board, move)
            if result['game_over']:
                return {'winner': result['winner'], 'turns_played': turn + 1}
            
            #check for endgame. IDK if it's inefficient but last_moves are converted into space representation for this.
            result = self.game.check_endgame(
                                            last_moves=[decimal_to_space(last_move,self.game.n_rows,self.game.n_cols) for last_move in last_moves],
                                            last_move=move, 
                                            player=player.get_role(),
                                            n_moves=turn)
            if result['game_over']:
                return {'winner': result['winner'], 'turns_played': turn + 1}
            
            #update the list of last_moves
            if len(last_moves) == 8:
                last_moves.pop(0)
            last_moves.append(move)

            #If opponent has no more legal moves, return current player as winner
            opponent = ATK if player == DEF else DEF
            if self.game.legal_moves(board, opponent) == 0:
                return player.get_role()
        #if 150 rounds pass, return draw
        #I think this won't trigger, since check_endgame also makes a check to see if max rounds are reached
        return {'winner': DRAW, 'turns_played': self.turn_limit}
    
    def play_n_matches(self, playerA, playerB, n) -> dict:
        """
        Play n matches between playerA and playerB.
        Half of the matches will have playerA as attacker, and half will have playerB attack.
        If n is odd, then playerA will be attacker one more time than playerB
        
        Return
        ------
        Nested dict with following format:
        
        Outer level with 'A_atk' and 'B_atk' as keys, 
        each storing dicts with info about the games where either A or B was attacker
        
        Inner level: each outer dict stores 3 dicts with keys 'A_wins', 'B_wins', and 'draw'.
        The values for each of these dicts contains 'count' and 'avg_turns_played' for corresponding category.
        """
        stats = {
            'A_atk': {
                'A_wins': {
                    'count': 0,
                    'avg_turns_played': 0
                },
                'B_wins': {
                    'count': 0,
                    'avg_turns_played': 0
                },
                'draw': {
                    'count': 0,
                    'avg_turns_played': 0
                }
            },
            'B_atk': {
                'B_wins': {
                    'count': 0,
                    'avg_turns_played': 0
                },
                'A_wins': {
                    'count': 0,
                    'avg_turns_played': 0
                },
                'draw': {
                    'count': 0,
                    'avg_turns_played': 0
                }
            }
        }
        # play n/2 games with A as attacker, rounded up
        for i in range(ceil(n/2)):
            res = self.play_game(playerA, playerB)
            #record match winner and turns played (we take average later)
            if res['winner'] == 1:
                stats['A_atk']['A_wins']['count'] += 1
                stats['A_atk']['A_wins']['avg_turns_played'] += res['turns_played']
            elif res['winner'] == 0:
                stats['A_atk']['B_wins']['count'] += 1
                stats['A_atk']['B_wins']['avg_turns_played'] += res['turns_played']
            else:
                stats['A_atk']['draw']['count'] += 1
                stats['A_atk']['draw']['avg_turns_played'] += res['turns_played']
                    
        #divide total turns by number of games to get average turns for each category
        if stats['A_atk']['A_wins']['avg_turns_played'] > 0:
            stats['A_atk']['A_wins']['avg_turns_played'] /= stats['A_atk']['A_wins']['count']
        if stats['A_atk']['B_wins']['avg_turns_played'] > 0:
            stats['A_atk']['B_wins']['avg_turns_played'] /= stats['A_atk']['B_wins']['count']
        if stats['A_atk']['draw']['avg_turns_played'] > 0:
            stats['A_atk']['draw']['avg_turns_played'] /= stats['A_atk']['draw']['count']
            
        #play n/2 games with B as attacker, rounded down
        for i in range(n//2):
            res = self.play_game(playerB, playerA)
            #record match winner and turns played (we take average later)
            if res['winner'] == 1:
                stats['B_atk']['B_wins']['count'] += 1
                stats['B_atk']['B_wins']['avg_turns_played'] += res['turns_played']
            if res['winner'] == 0:
                stats['B_atk']['A_wins']['count'] += 1
                stats['B_atk']['A_wins']['avg_turns_played'] += res['turns_played']
            if res['winner'] == -1:
                stats['B_atk']['draw']['count'] += 1
                stats['B_atk']['draw']['avg_turns_played'] += res['turns_played']
                    
        #divide total turns by number of games to get average turns for each category
        if stats['B_atk']['A_wins']['avg_turns_played'] > 0:
            stats['B_atk']['A_wins']['avg_turns_played'] /= stats['B_atk']['A_wins']['count']
        if stats['B_atk']['B_wins']['avg_turns_played'] > 0:
            stats['B_atk']['B_wins']['avg_turns_played'] /= stats['B_atk']['B_wins']['count']
        if stats['B_atk']['draw']['avg_turns_played'] > 0:
            stats['B_atk']['draw']['avg_turns_played'] /= stats['B_atk']['draw']['count']
        
        return stats
    
    def compare_2_individuals(self, netA, netB, n=100, epsilon=0.05) -> None:
        """
        Make players from 2 input networks, play n matches and print results
        """
        playerA = Player(netA, self.game, epsilon=epsilon)
        playerB = Player(netB, self.game, epsilon=epsilon)
        
        stats = self.play_n_matches(playerA, playerB, n)
        
        #print overall match stats
        print()
        print(f"Matches won by A: {stats['A_atk']['A_wins']['count'] + stats['B_atk']['A_wins']['count']} / {n}")
        print(f" - Average turns played: {(stats['A_atk']['A_wins']['avg_turns_played'] + stats['B_atk']['A_wins']['avg_turns_played']) / 2}")
        print(f"Matches won by B: {stats['A_atk']['B_wins']['count'] + stats['B_atk']['B_wins']['count']} / {n}")
        print(f" - Average turns played: {(stats['A_atk']['B_wins']['avg_turns_played'] + stats['B_atk']['B_wins']['avg_turns_played']) / 2}")
        print(f"Matches ending in draw: {stats['A_atk']['draw']['count'] + stats['B_atk']['draw']['count']} / {n}")
        print(f" - Average turns played: {(stats['A_atk']['draw']['avg_turns_played'] + stats['B_atk']['draw']['avg_turns_played']) / 2}")
        print()
        #print match stats for when A was attacker
        print( " * * * When A was attacker * * * ")
        print()
        print(f"Matches won by A: {stats['A_atk']['A_wins']['count']} / {ceil(n / 2)}")
        print(f" - Average turns played: {stats['A_atk']['A_wins']['avg_turns_played']}")
        print(f"Matches won by B: {stats['A_atk']['B_wins']['count']} / {ceil(n / 2)}")
        print(f" - Average turns played: {stats['A_atk']['B_wins']['avg_turns_played']}")
        print(f"Matches ending in draw: {stats['A_atk']['draw']['count']} / {ceil(n / 2)}")
        print(f" - Average turns played: {stats['A_atk']['draw']['avg_turns_played']}")
        print()
        #print match stats for when A was attacker
        print( " * * * When B was attacker * * * ")
        print()
        print(f"Matches won by A: {stats['B_atk']['A_wins']['count']} / {ceil(n / 2)}")
        print(f" - Average turns played: {stats['B_atk']['A_wins']['avg_turns_played']}")
        print(f"Matches won by B: {stats['B_atk']['B_wins']['count']} / {ceil(n / 2)}")
        print(f" - Average turns played: {stats['B_atk']['B_wins']['avg_turns_played']}")
        print(f"Matches ending in draw: {stats['B_atk']['draw']['count']} / {ceil(n / 2)}")
        print(f" - Average turns played: {stats['B_atk']['draw']['avg_turns_played']}")
        print()

        
        
            

def eval_genomes(genomes, config):
    #Initialize Tafl game engine
    game = GameEngine('gym_tafl/variants/custom.ini')
    
    players = []
    for i, genome_tuple in enumerate(genomes):
        players.append((i, Player(neat.nn.FeedForwardNetwork.create(genome_tuple[1], config), game, epsilon=0.1)))
    
    tournament = PureTournament(game=game)
    
    tournament_stats = tournament.play_tournament(players)
    
    for i, genome_tuple in enumerate(genomes):
        genome_tuple[1].fitness = tournament_stats['scores'][i]
        
    print("- - - Tournament Results - - -")
    print(f"Number of attacker victories: {tournament_stats['attacker_win_count']}")
    print(f"Number of defender victories: {tournament_stats['defender_win_count']}")
    print(f"Number of draws: {tournament_stats['draw_count']}")

def run(config_file):
    # Load configuration.
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         config_file)

    # Create the population, which is the top-level object for a NEAT run.
    p = neat.Population(config)

    # Add a stdout reporter to show progress in the terminal.
    p.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    p.add_reporter(stats)
    p.add_reporter(neat.Checkpointer(10))
    
    basic_player = p.run(eval_genomes, 5)

    # # Run for up to n generations.
    # surviving_player = p.run(eval_genomes, 100)

    # # Display the winning genome.
    # print('\nBest genome:\n{!s}'.format(surviving_player))

    # # Show output of the most fit genome against training data.
    # print('\nOutput:')
    # surviving_net = neat.nn.FeedForwardNetwork.create(surviving_player, config)
    # basic_net = neat.nn.FeedForwardNetwork.create(basic_player, config)
    # dump(surviving_net,open("best.pickle", "wb"))
    
    # print("= = = Gen 1 winner (A) VS Gen 100 winner (B) = = =")
    # tournament = PureTournament(game=GameEngine('gym_tafl/variants/custom.ini'))
    # tournament.compare_2_individuals(basic_net, surviving_net)

    # p = neat.Checkpointer.restore_checkpoint('neat-checkpoint-4')
    # p.run(eval_genomes, 10)

def run(config_file):
    # Load configuration.
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         config_file)

    # Create the population, which is the top-level object for a NEAT run.
    p = neat.Population(config)

    # Add a stdout reporter to show progress in the terminal.
    p.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    p.add_reporter(stats)
    p.add_reporter(neat.Checkpointer(10))
    
    basic_player = p.run(eval_genomes, 5)

    # # Run for up to n generations.
    # surviving_player = p.run(eval_genomes, 100)

    # # Display the winning genome.
    # print('\nBest genome:\n{!s}'.format(surviving_player))

    # # Show output of the most fit genome against training data.
    # print('\nOutput:')
    # surviving_net = neat.nn.FeedForwardNetwork.create(surviving_player, config)
    # basic_net = neat.nn.FeedForwardNetwork.create(basic_player, config)
    # dump(surviving_net,open("best.pickle", "wb"))
    
    # print("= = = Gen 1 winner (A) VS Gen 100 winner (B) = = =")
    # tournament = PureTournament(game=GameEngine('gym_tafl/variants/custom.ini'))
    # tournament.compare_2_individuals(basic_net, surviving_net)

    # p = neat.Checkpointer.restore_checkpoint('neat-checkpoint-4')
    # p.run(eval_genomes, 10)


if __name__ == '__main__':
    # Determine path to configuration file. This path manipulation is
    # here so that the script will run successfully regardless of the
    # current working directory.
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, 'config-feedforward.txt')
    run(config_path)

