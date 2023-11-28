import numpy as np
from random import randint, choice

class Player():
    """
    Class representing single TAFL player

    epsilon: exploration-exploitation parameter
    net: neat.nn.FeedForwardNetwork object storing one genome's NN
    game: GameEngine object for this player
    role: 1 if this player is an attacker, 0 otherwise
    """
    def __init__(self, epsilon, net, game, role):
        self.epsilon = epsilon
        self.net = net
        self.game = game
        self.role = role

    def choose_move(self, board, last_moves) -> int:
        """
        Get the action taken by the player at the current state of the given game.
    
        board: np array of the current game board
        last_moves: a list of integer representations of the previous moves

        Return
        ------
        The chosen action as an int
        """
        #copy all but the oldest move from last_moves
        last_moves_copy = [move for move in last_moves[1:]]
        #get legal moves from engine
        moves = self.game.legal_moves(board, self.role)
        #make random choice with probability epsilon
        if randint(0, 1) < self.epsilon:
            return choice(moves)
        
        #run the player's network on the successor from each action in order to find the best one
        bestMove = moves[0]
        bestScore = float('-inf')
        for move in moves:
            #make copy of current board to test each move on
            currBoardCopy = np.copy(board)
            self.game.alt_apply_move(currBoardCopy, move)

            #update the last_moves list with the current move being checked
            last_moves_copy.append(move)
            #network input is 49 ints from current board state + 8 ints representing last moves + 1 int representing current role
            networkInput = np.append(currBoardCopy.flatten(), last_moves_copy, self.role)
            currScore = self.net.activate(networkInput)

            if currScore > bestScore:
                bestScore = currScore
                bestMove = move

            #remove current move from last moves list
            last_moves_copy.pop(-1)
                
        return bestMove
    
