import numpy as np
import matplotlib
import squarify
from matplotlib import pyplot as plt
from game_implementation import Board_3D

# Set seed
np.random.seed(5)

# Do 10000 iterations
n_iter = 10000
final_scores = []
largest_tiles = []

for i in range(n_iter):
    # Set up the game
    g = Board_3D(rows=3, cols=3, pipes=3, prob_2=0.9, finish_value=2048)

    while True:
        # Get random input
        direction = np.random.randint(6)

        # Process move
        new_state, reward, done, info = g.step(direction)

        # Check termination
        if done:
            final_scores.append(g.score)
            largest_tiles.append(np.max(g.board))
            break


# Plot scores
plt.hist(final_scores, bins=30)
plt.xlabel('Score')
plt.ylabel('Frequency')
plt.title('Repeated Random Games')
plt.show()


# Get tile counts
tiles_achieved = [int(j) for j in sorted(list(set(largest_tiles)))]
tile_counts = [sum(value == np.array(largest_tiles)) for value in tiles_achieved]

# Create color palette, mapped to tile values
cmap = matplotlib.cm.Blues
norm = matplotlib.colors.Normalize(vmin=np.log2(tiles_achieved[0]), vmax=np.log2(tiles_achieved[-1]))
colors = [cmap(norm(np.log2(value))) for value in tiles_achieved]

# Plot largest tiles
fig, ax = plt.subplots(figsize=(6,6))
ax.set_axis_off()
ax.set_title('Largest Tiles')

# Add treemap
squarify.plot(
   sizes=tile_counts,
   label=tiles_achieved,
   ax=ax,
   color=colors
)
plt.show()

