import gym
import pygame
import numpy as np
from gym import spaces
from checkers import Game

# Color definitions
RED = (255, 0, 0)
BLUE = (0, 0, 255)
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
GOLD = (255, 215, 0)

class EnvCheckers(gym.Env):
    """
    Build an environment for checkers with the OpenAI Gym to make
    test and train efficient.
    """
    metadata = {'render.modes': ['human']}

    def __init__(self):
        super(EnvCheckers, self).__init__()
        self.game = Game()
        self.turn = BLUE  # Commence avec les bleus
        # Define the action space as a move between two squares on the 8x8 board
        self.action_space = spaces.Discrete(64 * 64)  # 64 start positions * 64 end positions
        # Observation space is a 8x8 grid representing the board state
        self.observation_space = spaces.Box(low=0, high=4, shape=(8, 8), dtype=np.int8)  # 0 for empty, 1 for red, 2 for blue, +2 for kings
        self.reset()
    
    def reset(self):
        """Reset the game to the initial state."""
        self.game = Game()
        self.game.setup()
        self.turn = BLUE 
        return self._get_obs() 
    
    def _get_obs(self):
        """Return the current board state as an observation."""
        board_matrix = np.zeros((8, 8), dtype=np.int8)
        for x in range(8):
            for y in range(8):
                piece = self.game.board.matrix[x][y].occupant
                if piece is not None:
                    if piece.color == RED:
                        board_matrix[x][y] = 1  # 1 for red
                    elif piece.color == BLUE:
                        board_matrix[x][y] = 2  # 2 for blue
                    if piece.king:
                        board_matrix[x][y] += 2  # +2 for kings
        
        return board_matrix

    def _get_legal_moves(self, start_pos):
        """Return all legal moves for a given position, considering mandatory captures."""
        legal_moves = self.game.board.legal_moves(start_pos)
        capture_moves = []
        for move in legal_moves:
            if abs(move[0] - start_pos[0]) > 1: 
                capture_moves.append(move)
        
        return capture_moves if capture_moves else legal_moves

    def step(self, action):
        """Apply an action and return the next state, reward, done, and info."""
        start_idx = action // 64
        end_idx = action % 64

        start_pos = (start_idx // 8, start_idx % 8)
        end_pos = (end_idx // 8, end_idx % 8)

        legal_moves = self._get_legal_moves(start_pos)
        if end_pos in legal_moves:
            self.game.board.move_piece(start_pos, end_pos)

            if abs(start_pos[0] - end_pos[0]) > 1:
                captured_pos = ((start_pos[0] + end_pos[0]) // 2, (start_pos[1] + end_pos[1]) // 2)
                self.game.board.remove_piece(captured_pos)

            self.turn = RED if self.turn == BLUE else BLUE

        done = self.game.check_for_endgame()
        reward = self._compute_reward(done)
        obs = self._get_obs()

        return obs, reward, done, {}

    def _compute_reward(self, done):
        """Compute the reward based on the game state."""
        if done:
            if self.turn == RED:
                return 1 
            else:
                return -1  
        return 0  
    
    def render(self, mode='human'):
        """Render the current state of the game."""
        self.game.graphics.update_display(self.game.board, self.game.selected_legal_moves, self.game.selected_piece)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.close()
    
    def close(self):
        """Close the game environment."""
        pygame.quit()


def main():
    env = EnvCheckers()
    obs = env.reset()
    done = False
    while not done:
        env.render()
        action = env.action_space.sample()
        obs, reward, done, info = env.step(action)
        print("obs: ", obs)
        print("reward: ", reward)
        print("done: ", done)
        print("info: ", info)

if __name__ == "__main__":
    main()