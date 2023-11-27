from gym_tafl.envs._game_engine import *

if __name__ == '__main__':
    game = GameEngine('gym_tafl/variants/custom.ini')
    board = np.zeros((game.n_rows,game.n_cols)) 
    game.fill_board(board)
    # print(game.board)
    print(board)
    print(game.legal_moves(board,ATK))
    moves = game.legal_moves(board,ATK)
    print(f"{moves[0]}: {decimal_to_space(moves[0],game.n_rows,game.n_cols)}")
    game.apply_move(board,decimal_to_space(moves[0],game.n_rows,game.n_cols))
    print(board)
    print(game.get_random_state())