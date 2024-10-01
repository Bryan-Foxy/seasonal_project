[Link of explained ressources](https://blog.paperspace.com/building-a-checkers-gaming-agent-using-neural-networks-and-reinforcement-learning/)

---
Tuesday 24/Sept/2024
---
Today, I updated the checkers environment to add two boxes that will contain information about the game.

- **Box 1**: For general information (Time elapsed, Players, Current player, Rules, Game board size, Round)
- **Box 2**: This box will be more specific, containing details about the game's results and the AI model powering the game. We will track new metrics such as Player 1's time and Player 2's time, pieces remaining on the board, win probability, prediction of remaining rounds to conclude the game, and the player most likely to win.

---
Wednesday 26/Sept/2024
---
I update the box 1, I make it more dynamic like real time elapsed, round update automaticily also the current player.

---
Friday 27/Sept/2024
---
I update the gym env cuz, I update the reward gestion.

---
30/09/2024
---
I update rules, now the env takes position to move, and update the reward gestion to be more efficient, also I created a logs folder to store metadata about the lauching of the env
