import gym
import pygame
import numpy as np
from gym import spaces
from checkers import Game
from functions import get_metrics, draw_box1
import uuid
import os
import json

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
        self.total_reward = 0
        self.step_rewards = []
        
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
        self.total_reward = 0  # Reset total reward
        self.step_rewards = []  # Clear previous rewards
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

    def _get_mandatory_captures(self):
        """Check if there are mandatory captures for the current player."""
        mandatory_captures = []
        for x in range(8):
            for y in range(8):
                piece = self.game.board.matrix[x][y].occupant
                if piece and piece.color == self.turn:
                    legal_moves = self._get_legal_moves((x, y))
                    # Filter only capture moves
                    capture_moves = [move for move in legal_moves if abs(move[0] - x) > 1]
                    if capture_moves:
                        mandatory_captures.append((x, y, capture_moves))
        return mandatory_captures

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
        
        # Vérification des captures obligatoires avant de continuer
        mandatory_captures = self._get_mandatory_captures()
        
        # Si des captures sont obligatoires et que l'action n'est pas une capture
        if mandatory_captures:
            for capture in mandatory_captures:
                if start_pos == (capture[0], capture[1]) and end_pos in capture[2]:
                    break
            else:
                print("Une capture est obligatoire!")
                reward -= 5  # Pénalité pour ne pas avoir effectué la capture obligatoire
                self.step_rewards.append(reward)
                return self._get_obs(), reward, False, {}

        if end_pos in legal_moves:
            self.game.board.move_piece(start_pos, end_pos)

            # Si un pion a été capturé
            if abs(start_pos[0] - end_pos[0]) > 1:
                captured_pos = ((start_pos[0] + end_pos[0]) // 2, (start_pos[1] + end_pos[1]) // 2)
                self.game.board.remove_piece(captured_pos)
                reward += 10  # Récompense pour avoir capturé un pion
            else:
                reward += 1  # Récompense pour avoir effectué un mouvement valide sans capture

            # Alterner les tours
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

        self.total_reward += reward
        self.step_rewards.append(reward)
        obs = self._get_obs()

        return obs, reward, done, {}

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
    
    def save_logs(self):
        """Save logs in a folder with a unique identifier."""
        log_data = {
            "total_reward": self.total_reward,
            "step_rewards": self.step_rewards,
            "round": self.round,
            "winner": "RED" if self.turn == RED else "BLUE"
        }
        # Generate a unique file name using UUID
        log_filename = str(uuid.uuid4()) + ".json"
        log_dir = "logs"
        os.makedirs(log_dir, exist_ok=True)  # Create the logs directory if it doesn't exist
        log_filepath = os.path.join(log_dir, log_filename)

        # Save log data to a JSON file
        with open(log_filepath, 'w') as log_file:
            json.dump(log_data, log_file, indent=4)
        
        print(f"Logs saved to {log_filepath}")

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

    # Save logs at the end of the game
    env.save_logs()

if __name__ == "__main__":
    main()