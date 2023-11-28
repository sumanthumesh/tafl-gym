import numpy as np
from random import randint, choice

class Player():
    """
    Class to represent a single TAFL player

    epsilon: exploration-exploitation parameter
    net: neat.nn.FeedForwardNetwork object storing one genome's NN
    game: GameEngine object for this player
    """
    def __init__(self, epsilon, net, game, role):
        self.epsilon = epsilon,
        self.net = net,
        self.game = game,
        self.role = role
        self.last_moves = []

    def choose_move(self, board):
        """
        Get the action taken by the player at the current state of the given game.
    
        board: np array of the current game board

        Return
        ------
        The chosen action (as an int, so you need to use alt_apply_move on it)
        """
        moves = self.game.legal_moves(board, self.role)
        #make random choice with probability epsilon
        if randint(0, 1) < self.epsilon:
            return choice(moves)
        
        #run the player's network on the successor from each action in order to find the best one
        bestMove = -1
        bestScore = float('-inf')
        for move in moves:
            #make copy of current board to test each move on
            currBoardCopy = np.copy(board)
            move_info = self.game.alt_apply_move(currBoardCopy, move)
            #if the move results in a win, immediately return it
            if move_info['game_over']:
                return move
            #TODO: Create variable storing current state information needed for neural network 
            currState = 
            currScore = self.net.activate(currState)

            if currScore > bestScore:
                bestScore = currScore
                bestMove = move
        return bestMove

        # valids = self.game.getValidMoves(board, board.getPlayerToMove())
        # candidates = []
        # for a in range(self.game.getActionSize()):
        #     if valids[a]==0:
        #         continue
        #     nextBoard, _ = self.game.getNextState(board, board.getPlayerToMove(), a)
        #     score = self.game.getScore(nextBoard, board.getPlayerToMove())
        #     candidates += [(-score, a)]
        # candidates.sort()
        # return candidates[0][1]
    


        # game = GameEngine('gym_tafl/variants/custom.ini')
        # board = np.zeros((game.n_rows,game.n_cols)) 
        # game.fill_board(board)
        # # print(game.board)
        # print(board)
        # print(game.legal_moves(board,ATK))
        # moves = game.legal_moves(board,ATK)
        # print(f"{moves[0]}: {decimal_to_space(moves[0],game.n_rows,game.n_cols)}")
        # game.apply_move(board,decimal_to_space(moves[0],game.n_rows,game.n_cols))
        # print(board)
        # print(game.get_random_state())
    
