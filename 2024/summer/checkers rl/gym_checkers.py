import gym
import pygame
import numpy as np
from gym import spaces
from checkers import Game
from functions import get_metrics, draw_box1

# Color definitions
RED = (255, 0, 0)
BLUE = (0, 0, 255)
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
GOLD = (255, 215, 0)

class EnvCheckers(gym.Env):
    """
    A checkers environment for reinforcement learning using OpenAI Gym.
    """
    metadata = {'render.modes': ['human']}

    def __init__(self):
        super(EnvCheckers, self).__init__()
        self.game = Game()
        self.turn = BLUE
        self.round = 1
        self.start_time = pygame.time.get_ticks()  # Start the timer
        
        # Define the action space (64 start positions * 64 end positions)
        self.action_space = spaces.Discrete(64 * 64)
        
        # Define the observation space (8x8 board, values: 0=empty, 1=red, 2=blue, +2=kings)
        self.observation_space = spaces.Box(low=0, high=4, shape=(8, 8), dtype=np.int8)
        self.reset()

    def reset(self):
        """Reset the game to its initial state."""
        self.game = Game()
        self.game.setup()
        self.turn = BLUE
        self.round = 1
        return self._get_obs()

    def _get_obs(self):
        """Return the current board state as an observation."""
        board_matrix = np.zeros((8, 8), dtype=np.int8)
        for x in range(8):
            for y in range(8):
                piece = self.game.board.matrix[x][y].occupant
                if piece is not None:
                    if piece.color == RED:
                        board_matrix[x][y] = 1  # Red piece
                    elif piece.color == BLUE:
                        board_matrix[x][y] = 2  # Blue piece
                    if piece.king:
                        board_matrix[x][y] += 2  # King piece
        return board_matrix

    def _get_legal_moves(self, start_pos):
        """Return all legal moves for a given position, considering mandatory captures."""
        legal_moves = self.game.board.legal_moves(start_pos)
        capture_moves = [move for move in legal_moves if abs(move[0] - start_pos[0]) > 1]  # Capture moves

        # If capture moves exist, return only those (capture is mandatory)
        return capture_moves if capture_moves else legal_moves

    def step(self, action):
        """Apply an action and return the next state, reward, done, and info."""
        start_idx = action // 64
        end_idx = action % 64

        start_pos = (start_idx // 8, start_idx % 8)
        end_pos = (end_idx // 8, end_idx % 8)

        print("Coordonnées de départ :", start_pos)
        print("Coordonnées d'arrivée :", end_pos)

        legal_moves = self._get_legal_moves(start_pos)

        reward = 0  # Initialisation de la récompense

        if end_pos in legal_moves:
            self.game.board.move_piece(start_pos, end_pos)

            # Si un pion a été capturé
            if abs(start_pos[0] - end_pos[0]) > 1:
                captured_pos = ((start_pos[0] + end_pos[0]) // 2, (start_pos[1] + end_pos[1]) // 2)
                self.game.board.remove_piece(captured_pos)
                reward += 10  # Récompense pour avoir capturé un pion
            else:
                reward += 1  # Récompense pour avoir effectué un mouvement valide sans capture

            self.turn = RED if self.turn == BLUE else BLUE
            self.round += 0.5

        else:
            print("Mouvement illégal!")
            reward -= 5  # Pénalité pour mouvement illégal

        # Vérifier la fin de la partie
        done = self.game.check_for_endgame()

        # Si la partie est terminée, ajouter une récompense/pénalité
        if done:
            if self.turn == RED:
                reward -= 20  # Pénalité pour le joueur Bleu s'il a perdu
            else:
                reward += 20  # Récompense pour le joueur Bleu s'il a gagné

        obs = self._get_obs()

        return obs, reward, done, {}

    def _handle_mouse_events(self):
        """Handle mouse events to simulate user input."""
        mouse_pos = self.game.graphics.board_coords(pygame.mouse.get_pos())
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.close()
            elif event.type == pygame.MOUSEBUTTONDOWN:
                if self.game.hop is False:
                    if self.game.board.location(mouse_pos).occupant and self.game.board.location(mouse_pos).occupant.color == self.turn:
                        self.game.selected_piece = mouse_pos
                    elif self.game.selected_piece and mouse_pos in self.game.board.legal_moves(self.game.selected_piece):
                        self.game.board.move_piece(self.game.selected_piece, mouse_pos)
                        if mouse_pos not in self.game.board.adjacent(self.game.selected_piece):
                            captured_pos = ((self.game.selected_piece[0] + mouse_pos[0]) // 2, (self.game.selected_piece[1] + mouse_pos[1]) // 2)
                            self.game.board.remove_piece(captured_pos)
                            self.game.hop = True
                            self.game.selected_piece = mouse_pos
                        else:
                            self.end_turn()
                elif self.game.hop is True:
                    if self.game.selected_piece and mouse_pos in self.game.board.legal_moves(self.game.selected_piece, self.game.hop):
                        self.game.board.move_piece(self.game.selected_piece, mouse_pos)
                        captured_pos = ((self.game.selected_piece[0] + mouse_pos[0]) // 2, (self.game.selected_piece[1] + mouse_pos[1]) // 2)
                        self.game.board.remove_piece(captured_pos)
                    if not self.game.board.legal_moves(mouse_pos, self.game.hop):
                        self.end_turn()
                    else:
                        self.game.selected_piece = mouse_pos

    def _compute_reward(self, done):
        """Compute the reward based on the game state."""
        if done:
            if self.turn == RED:
                return 1  # Reward for Red win
            else:
                return -1  # Reward for Blue win
        return 0  # No reward if the game is still ongoing
    
    def render(self, mode='human'):
        """Render the current state of the game."""
        self.game.graphics.update_display(self.game.board, self.game.selected_legal_moves, self.turn, self.round, self.game.selected_piece)
        
        # Box 1: General game information (round, turn, etc.)
        draw_box1(self.game.graphics.screen, self.game.graphics.titlefont, self.game.graphics.font, self.start_time, self.game.graphics.window_size, self.round, self.turn, WHITE, BLUE, RED, BLACK)
        
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
        print("action", action)
        obs, reward, done, info = env.step(action)
        print("obs: ", obs)
        print("reward: ", reward)
        print("done: ", done)
        print("info: ", info)

if __name__ == "__main__":
    main()