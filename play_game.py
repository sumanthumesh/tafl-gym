from gym_tafl.envs._game_engine import *
from gym_tafl.players import Player

if __name__ == '__main__':
    game = GameEngine('gym_tafl/variants/custom.ini')
    player1 = Player(0, None, game, DEF)
    board = np.zeros((game.n_rows,game.n_cols)) 
    game.fill_board(board)
    # print(game.board)
    #this code runs through a player choosing a move
    # last_moves = [0 for i in range(8)]
    # print(last_moves)
    # print(player1.choose_move(board, last_moves))
    # print(last_moves)

    # print(board)
    # print(game.legal_moves(board,1))
    # moves = game.legal_moves(board,ATK)
    # print(f"{moves[0]}: {decimal_to_space(moves[0],game.n_rows,game.n_cols)}")
    # game.apply_move(board,decimal_to_space(moves[0],game.n_rows,game.n_cols))
    # print(board)
    # print(game.get_random_state())