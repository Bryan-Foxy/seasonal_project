"""
checkers.py

A simple checkers engine written in Python with the pygame 1.9.1 libraries.

Here are the rules I am using: http://boardgames.about.com/cs/checkersdraughts/ht/play_checkers.htm

I adapted some code from checkers.py found at 
http://itgirl.dreamhosters.com/itgirlgames/games/Program%20Leaders/ClareR/Checkers/checkers.py starting on line 159 of my program.

This is the final version of my checkers project for Programming Workshop at Marlboro College. The entire thing has been rafactored and made almost completely object oriented.

Funcitonalities include:

- Having the pieces and board drawn to the screen

- The ability to move pieces by clicking on the piece you want to move, then clicking on the square you would
  like to move to. You can change you mind about the piece you would like to move, just click on a new piece of yours.

- Knowledge of what moves are legal. When moving pieces, you'll be limited to legal moves.

- Capturing

- DOUBLE capturing etc.

- Legal move and captive piece highlighting

- Turn changes

- Automatic kinging and the ability for them to move backwords

- Automatic check for and end game. 

- A silky smoooth 60 FPS!

Everest Witman - May 2014 - Marlboro College - Programming Workshop 
"""
import math
import pygame, sys
from pygame.locals import *
from functions import *

pygame.font.init()

##COLORS##
#             R    G    B 
WHITE    = (255, 255, 255)
BLUE     = (  0,   0, 255)
RED      = (255,   0,   0)
BLACK    = (  0,   0,   0)
GOLD     = (255, 215,   0)
HIGH     = (160, 190, 255)

##DIRECTIONS##
NORTHWEST = "northwest"
NORTHEAST = "northeast"
SOUTHWEST = "southwest"
SOUTHEAST = "southeast"

class Game:
	"""
	The main game control.
	"""

	def __init__(self):
		self.graphics = Graphics()
		self.board = Board()
		
		self.turn = BLUE
		self.round_number = 1
		self.selected_piece = None # a board location. 
		self.hop = False
		self.selected_legal_moves = []

	def setup(self):
		"""Draws the window and board at the beginning of the game"""
		self.graphics.setup_window()

	def event_loop(self, action=None):
		"""
		Boucle d'événements. Cela permet de déclencher des événements comme
		un clic de souris ou un mouvement direct via les coordonnées dans Gym.
		
		Paramètre:
			action (tuple): Tuple d'action directe sous la forme (start_pos, end_pos).
							Si aucune action n'est donnée, les événements de la souris sont gérés.
		"""
		if action:
			start_pos, end_pos = action
			self.move_piece_direct(start_pos, end_pos)
		else:
			self.mouse_pos = self.graphics.board_coords(pygame.mouse.get_pos())  # quelle case la souris survole-t-elle ?
			if self.selected_piece is not None:
				self.selected_legal_moves = self.board.legal_moves(self.selected_piece, self.hop)

			for event in pygame.event.get():
				if event.type == QUIT:
					self.terminate_game()

				if event.type == MOUSEBUTTONDOWN:
					if not self.hop:
						if self.board.location(self.mouse_pos).occupant and self.board.location(self.mouse_pos).occupant.color == self.turn:
							self.selected_piece = self.mouse_pos
							# Si des captures sont disponibles, les forcer
							legal_moves = self.board.legal_moves(self.mouse_pos)
							if any(abs(move[0] - self.mouse_pos[0]) > 1 for move in legal_moves):
								self.selected_legal_moves = legal_moves

						elif self.selected_piece and self.mouse_pos in self.selected_legal_moves:
							self.board.move_piece(self.selected_piece, self.mouse_pos)

							if self.mouse_pos not in self.board.adjacent(self.selected_piece):
								self.board.remove_piece(((self.selected_piece[0] + self.mouse_pos[0]) >> 1, (self.selected_piece[1] + self.mouse_pos[1]) >> 1))
								self.hop = True
								self.selected_piece = self.mouse_pos
							else:
								self.end_turn()

					if self.hop:
						if self.selected_piece and self.mouse_pos in self.board.legal_moves(self.selected_piece, self.hop):
							self.board.move_piece(self.selected_piece, self.mouse_pos)
							self.board.remove_piece(((self.selected_piece[0] + self.mouse_pos[0]) >> 1, (self.selected_piece[1] + self.mouse_pos[1]) >> 1))

						if not self.board.legal_moves(self.mouse_pos, self.hop):
							self.end_turn()
						else:
							self.selected_piece = self.mouse_pos


	def update(self):
		"""Calls on the graphics class to update the game display."""
		self.graphics.update_display(self.board, self.selected_legal_moves, self.selected_piece, self.round_number, self.turn)

	def terminate_game(self):
		"""Quits the program and ends the game."""
		pygame.quit()
		sys.exit

	def main(self):
		""""This executes the game and controls its flow."""
		self.setup()

		while True: # main game loop
			self.event_loop()
			self.update()

	def end_turn(self):
		"""
		End the turn. Switches the current player. 
		end_turn() also checks for and game and resets a lot of class attributes.
		"""
		if self.turn == BLUE:
			self.turn = RED
		else:
			self.turn = BLUE
		self.round_number += 0.5

		self.selected_piece = None
		self.selected_legal_moves = []
		self.hop = False

		if self.check_for_endgame():
			if self.turn == BLUE:
				self.graphics.draw_message("RED WINS!")
			else:
				self.graphics.draw_message("BLUE WINS!")

	def check_for_endgame(self):
		"""
		Checks to see if a player has run out of moves or pieces. If so, then return True. Else return False.
		"""
		for x in range(8):
			for y in range(8):
				if self.board.location((x,y)).color == BLACK and self.board.location((x,y)).occupant != None and self.board.location((x,y)).occupant.color == self.turn:
					if self.board.legal_moves((x,y)) != []:
						return False

		return True
	
	def move_piece_direct(self, start_pos, end_pos):
		"""
		Move directly a piece with its coordinates
		"""
		legal_moves = self.board.legal_moves(start_pos)

		if end_pos in legal_moves:
			self.board.move_piece(start_pos, end_pos)

			if abs(start_pos[0] - end_pos[0]) > 1: 
				captured_pos = ((start_pos[0] + end_pos[0]) // 2, (start_pos[1] + end_pos[1]) // 2)
				self.board.remove_piece(captured_pos)

			if self.board.legal_moves(end_pos, self.hop) == []:
						self.end_turn()
			else:
				self.selected_piece = end_pos
				self.hop = True
		else:
			return False

class Graphics:
	def __init__(self):
		self.caption = "Checkers"

		self.fps = 60
		self.clock = pygame.time.Clock()

		self.window_size = 600
		self.titlefont = pygame.font.Font(None, 36)
		self.font = pygame.font.Font(None, 18)
		self.start_time = time.time()
		self.screen = pygame.display.set_mode((self.window_size + 200, self.window_size))
		self.background = pygame.image.load('resources/board.png')
		self.line = pygame.draw.line(surface = self.screen, color = WHITE, start_pos = (self.window_size, 0), end_pos = (self.window_size, self.window_size + 200), width = 2)
		#draw_box1(self.screen, self.titlefont, self.font, start_time, self.window_size, WHITE, BLUE, RED)
		#self.box1 = pygame.draw.rect(surface = self.screen, color = WHITE, rect = (self.window_size, 0, 200, 100), width = 2)
		#self.box2 = pygame.draw.rect(surface = self.screen, color = WHITE, rect = (self.window_size, 150, 200, 150), width = 2)


		self.square_size = self.window_size >> 3
		self.piece_size = self.square_size >> 1

		self.message = False

	def setup_window(self):
		"""
		This initializes the window and sets the caption at the top.
		"""
		pygame.init()
		pygame.display.set_caption(self.caption)

	def update_display(self, board, legal_moves, selected_piece, round_number, turn):
		"""
		This updates the current display.
		"""
		self.screen.blit(self.background, (0,0))
		draw_box1(self.screen, self.titlefont, self.font, self.start_time, self.window_size, round_number, turn, WHITE, BLUE, RED, BLACK)

		self.highlight_squares(legal_moves, selected_piece)
		self.draw_board_pieces(board)

		if self.message:
			self.screen.blit(self.text_surface_obj, self.text_rect_obj)

		pygame.display.update()
		self.clock.tick(self.fps)

	def draw_board_squares(self, board):
		"""
		Takes a board object and draws all of its squares to the display
		"""
		for x in range(8):
			for y in range(8):
				pygame.draw.rect(self.screen, board[x][y].color, (x * self.square_size, y * self.square_size, self.square_size, self.square_size), )
	
	def draw_board_pieces(self, board):
		"""
		Takes a board object and draws all of its pieces to the display
		"""
		for x in range(8):
			for y in range(8):
				if board.matrix[x][y].occupant != None:
					pygame.draw.circle(self.screen, board.matrix[x][y].occupant.color, self.pixel_coords((x,y)), self.piece_size) 

					if board.location((x,y)).occupant.king == True:
						pygame.draw.circle(self.screen, GOLD, self.pixel_coords((x,y)), int (self.piece_size / 1.7), self.piece_size >> 2)


	def pixel_coords(self, board_coords):
		"""
		Takes in a tuple of board coordinates (x,y) 
		and returns the pixel coordinates of the center of the square at that location.
		"""
		return (board_coords[0] * self.square_size + self.piece_size, board_coords[1] * self.square_size + self.piece_size)

	def board_coords(self, pixel):
		"""
		Does the reverse of pixel_coords(). Takes in a tuple of of pixel coordinates and returns what square they are in.
		"""
		return (pixel[0] // self.square_size, pixel[1] // self.square_size)

	def highlight_squares(self, squares, origin):
		"""
		Squares is a list of board coordinates. 
		highlight_squares highlights them.
		"""
		for square in squares:
			pygame.draw.rect(self.screen, HIGH, (square[0] * self.square_size, square[1] * self.square_size, self.square_size, self.square_size))	

		if origin != None:
			pygame.draw.rect(self.screen, HIGH, (origin[0] * self.square_size, origin[1] * self.square_size, self.square_size, self.square_size))

	def draw_message(self, message):
		"""
		Draws message to the screen. 
		"""
		self.message = True
		self.font_obj = pygame.font.Font('freesansbold.ttf', 44)
		self.text_surface_obj = self.font_obj.render(message, True, HIGH, BLACK)
		self.text_rect_obj = self.text_surface_obj.get_rect()
		self.text_rect_obj.center = (self.window_size >> 1, self.window_size >> 1)

class Board:
	def __init__(self):
		self.matrix = self.new_board()

	def new_board(self):
		"""
		Create a new board matrix.
		"""

		# initialize squares and place them in matrix

		matrix = [[None] * 8 for i in range(8)]

		# The following code block has been adapted from
		# http://itgirl.dreamhosters.com/itgirlgames/games/Program%20Leaders/ClareR/Checkers/checkers.py
		for x in range(8):
			for y in range(8):
				if (x % 2 != 0) and (y % 2 == 0):
					matrix[y][x] = Square(WHITE)
				elif (x % 2 != 0) and (y % 2 != 0):
					matrix[y][x] = Square(BLACK)
				elif (x % 2 == 0) and (y % 2 != 0):
					matrix[y][x] = Square(WHITE)
				elif (x % 2 == 0) and (y % 2 == 0): 
					matrix[y][x] = Square(BLACK)

		# initialize the pieces and put them in the appropriate squares

		for x in range(8):
			for y in range(3):
				if matrix[x][y].color == BLACK:
					matrix[x][y].occupant = Piece(RED)
			for y in range(5, 8):
				if matrix[x][y].color == BLACK:
					matrix[x][y].occupant = Piece(BLUE)

		return matrix

	def board_string(self, board):
		"""
		Takes a board and returns a matrix of the board space colors. Used for testing new_board()
		"""

		board_string = [[None] * 8] * 8 

		for x in range(8):
			for y in range(8):
				if board[x][y].color == WHITE:
					board_string[x][y] = "WHITE"
				else:
					board_string[x][y] = "BLACK"


		return board_string
	
	def rel(self, dir, pixel):
		"""
		Returns the coordinates one square in a different direction to (x,y).

		===DOCTESTS===

		>>> board = Board()

		>>> board.rel(NORTHWEST, (1,2))
		(0,1)

		>>> board.rel(SOUTHEAST, (3,4))
		(4,5)

		>>> board.rel(NORTHEAST, (3,6))
		(4,5)

		>>> board.rel(SOUTHWEST, (2,5))
		(1,6)
		"""
		x = pixel[0]
		y = pixel[1]
		if dir == NORTHWEST:
			return (x - 1, y - 1)
		elif dir == NORTHEAST:
			return (x + 1, y - 1)
		elif dir == SOUTHWEST:
			return (x - 1, y + 1)
		elif dir == SOUTHEAST:
			return (x + 1, y + 1)
		else:
			return 0

	def adjacent(self, pixel):
		"""
		Returns a list of squares locations that are adjacent (on a diagonal) to (x,y).
		"""
		x = pixel[0]
		y = pixel[1]

		return [self.rel(NORTHWEST, (x,y)), self.rel(NORTHEAST, (x,y)),self.rel(SOUTHWEST, (x,y)),self.rel(SOUTHEAST, (x,y))]

	def location(self, pixel):
		"""
		Takes a set of coordinates as arguments and returns self.matrix[x][y]
		This can be faster than writing something like self.matrix[coords[0]][coords[1]]
		"""
		x = pixel[0]
		y = pixel[1]

		return self.matrix[x][y]

	def blind_legal_moves(self, pixel):
		"""
		Returns a list of blind legal move locations from a set of coordinates (x,y) on the board. 
		If that location is empty, then blind_legal_moves() return an empty list.
		"""

		x = pixel[0]
		y = pixel[1]
		if self.matrix[x][y].occupant != None:
			
			if self.matrix[x][y].occupant.king == False and self.matrix[x][y].occupant.color == BLUE:
				blind_legal_moves = [self.rel(NORTHWEST, (x,y)), self.rel(NORTHEAST, (x,y))]
				
			elif self.matrix[x][y].occupant.king == False and self.matrix[x][y].occupant.color == RED:
				blind_legal_moves = [self.rel(SOUTHWEST, (x,y)), self.rel(SOUTHEAST, (x,y))]

			else:
				blind_legal_moves = [self.rel(NORTHWEST, (x,y)), self.rel(NORTHEAST, (x,y)), self.rel(SOUTHWEST, (x,y)), self.rel(SOUTHEAST, (x,y))]

		else:
			blind_legal_moves = []

		return blind_legal_moves

	def legal_moves(self, pixel, hop=False):
		"""
		Returns a list of legal move locations from a given set of coordinates (x,y) on the board.
		If that location is empty, then legal_moves() returns an empty list.
		"""
		x = pixel[0]
		y = pixel[1]
		blind_legal_moves = self.blind_legal_moves((x,y))
		legal_moves = []
		capture_moves = []

		for move in blind_legal_moves:
			if self.on_board(move):
				# Vérification des mouvements de capture
				if self.location(move).occupant and self.location(move).occupant.color != self.location((x,y)).occupant.color:
					capture_move = (move[0] + (move[0] - x), move[1] + (move[1] - y))
					if self.on_board(capture_move) and self.location(capture_move).occupant == None:
						capture_moves.append(capture_move)
				elif not self.location(move).occupant:  # Mouvements normaux
					legal_moves.append(move)
		
		# Si des mouvements de capture existent, les rendre obligatoires
		if capture_moves:
			return capture_moves
		
		return legal_moves

	def remove_piece(self, pixel):
		"""
		Removes a piece from the board at position (x,y). 
		"""
		x = pixel[0]
		y = pixel[1]
		self.matrix[x][y].occupant = None

	def move_piece(self, pixel_start, pixel_end):
		"""
		Move a piece from (start_x, start_y) to (end_x, end_y).
		"""
		start_x = pixel_start[0]
		start_y = pixel_start[1]
		end_x = pixel_end[0]
		end_y = pixel_end[1]

		self.matrix[end_x][end_y].occupant = self.matrix[start_x][start_y].occupant
		self.remove_piece((start_x, start_y))

		self.king((end_x, end_y))

	def is_end_square(self, coords):
		"""
		Is passed a coordinate tuple (x,y), and returns true or 
		false depending on if that square on the board is an end square.

		===DOCTESTS===

		>>> board = Board()

		>>> board.is_end_square((2,7))
		True

		>>> board.is_end_square((5,0))
		True

		>>>board.is_end_square((0,5))
		False
		"""

		if coords[1] == 0 or coords[1] == 7:
			return True
		else:
			return False

	def on_board(self, pixel):
		"""
		Checks to see if the given square (x,y) lies on the board.
		If it does, then on_board() return True. Otherwise it returns false.

		===DOCTESTS===
		>>> board = Board()

		>>> board.on_board((5,0)):
		True

		>>> board.on_board(-2, 0):
		False

		>>> board.on_board(3, 9):
		False
		"""

		x = pixel[0]
		y = pixel[1]
		if x < 0 or y < 0 or x > 7 or y > 7:
			return False
		else:
			return True


	def king(self, pixel):
		"""
		Takes in (x,y), the coordinates of square to be considered for kinging.
		If it meets the criteria, then king() kings the piece in that square and kings it.
		"""
		x = pixel[0]
		y = pixel[1]
		if self.location((x,y)).occupant != None:
			if (self.location((x,y)).occupant.color == BLUE and y == 0) or (self.location((x,y)).occupant.color == RED and y == 7):
				self.location((x,y)).occupant.king = True 

class Piece:
	def __init__(self, color, king = False):
		self.color = color
		self.king = king

class Square:
	def __init__(self, color, occupant = None):
		self.color = color # color is either BLACK or WHITE
		self.occupant = occupant # occupant is a Square object

def main():
	game = Game()
	game.main()

if __name__ == "__main__":
	main()