from game_implementation import Board_3D
from model_construction import DQNAgent
from training import load_checkpoint, train_agent
import matplotlib.pyplot as plt
import numpy as np

# Inputs
checkpoint_path = ''            # load saved agent: 'checkpoint_episode_###.pth'
episodes = 10                   # number of desired episodes, actual training used around 500 episodes
save_plots = False              # whether to save plots of results

# NOTE: Checkpoints save every 10 episodes, so files for episode 0 and episode 10 should exist once those episodes complete.
#       You might need to look around your local computer files in order to find them, but they are there.

if __name__ == "__main__":
    env = Board_3D(rows=3, cols=3, pipes=3)
    input_shape = (env.rows, env.cols, env.pipes)
    action_size = 6
    
    try:
        # Load parameters into model
        agent = DQNAgent(input_shape, action_size)
        start_episode, scores = load_checkpoint(agent, checkpoint_path)
        print(f"Loaded checkpoint from episode {start_episode}")
    except FileNotFoundError:
        print("No checkpoint found; starting fresh.")
        agent = None
        start_episode = 0
        scores = []

    # Continue training (or start fresh if no checkpoint was loaded)
    agent, scores, max_tiles = train_agent(env, episodes=episodes, agent=agent, start_episode=start_episode, sim=False)


# ===== PLOTTING =====

# Total Episode Score
fig1, ax1 = plt.subplots()
ax1.plot(range(len(scores)), scores, label='Score', color='blue')
ax1.set_xlabel('Episode')
ax1.set_ylabel('Score')
ax1.set_title('Score per Episode')
ax1.legend()
ax1.grid(True)  # Add grid lines for better readability


# Cumulative Average Score up to Episode
avg_scores = np.cumsum(scores) / np.arange(1, len(scores) + 1)

fig2, ax2 = plt.subplots()
ax2.plot(range(len(scores)), avg_scores, label='Average Score', color='green')
ax2.set_xlabel('Episode')
ax2.set_ylabel('Score')
ax2.set_title('Cumulative Average Score up to Episode')
ax2.legend()
ax2.grid(True)

# Maximum Tiles Achieved
fig3, ax3 = plt.subplots()
ax3.plot(range(len(scores)), max_tiles, label='Tile Value', color='red')
ax3.set_yscale('log', base=2)
ax3.set_xlabel('Episode')
ax3.set_ylabel('Tile Value')
ax3.set_title('Maximum Tiles Achieved')
ax3.legend()
ax3.grid(True)

# Display all three graphs
if save_plots: plt.savefig('training_plots.png')
plt.show()
