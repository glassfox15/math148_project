from game_implementation import Board_3D
import numpy as np

game = Board_3D(rows=3, cols=3, pipes=3, prob_2=0.9, finish_value=2048)
print("Initial board:")
print(game.board, '\n---------------\n')

direction_map = {'Q':0, 'W':1, 'E':2, 'A':3, 'S':4, 'D':5}

while True:
    # Get input
    direction = input('(Q) away | (W) up | (E) toward | (A) left | (S) down | (D) right | (X) exit: ').upper()

    if direction == 'X':
        print("Game exited by user")
        break

    if direction not in direction_map:
        print("Invalid direction! Use Q/W/E/A/S/D/X")
        continue

    # Process move
    action = direction_map[direction]
    new_state, reward, done, info = game.step(action)

    # Update display
    print(f"Move result ({direction}):")
    print(np.array(game.board, dtype=np.int32), '\n---------------\n')
    print(f"Score: {game.score}\n")

    # Check termination
    if done:
        print('Game Over! Final score:', game.score)
        break