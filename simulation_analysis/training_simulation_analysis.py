from game_implementation import Board_3D
from model_construction import *
from training import load_checkpoint, train_agent
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import torch

"""
This code simulates training runs of the model at different episodes.
The purpose is to understand possible and realized training behavior.

Requires the following PRE-TRAINED model checkpoints:
    checkpoint_episode_50.pth
    checkpoint_episode_100.pth
    checkpoint_episode_150.pth
    checkpoint_episode_190.pth
    checkpoint_episode_200.pth
    checkpoint_episode_250.pth
    checkpoint_episode_300.pth
    checkpoint_episode_340.pth
    checkpoint_episode_400.pth

A maximum of 9 checkpoints (0 will be run automatically, for a total of 10) can be visualized at once.

This code runs the training iteration WITHOUT updating parameters or updating checkpoints.
The final result is the simulated data and a set of plots displaying the simulations.
"""

save_data = False       # whether you want to store the data
save_plots = False      # whether you want to store the plots

# Prepare Simulations
n_sims = 1000
eps = [50, 100, 150, 170, 200, 250, 300, 350, 400] # episodes to load
ep_labs = ['ep 0', *['ep ' + str(e) for e in eps]] # episode labels

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

### ===== SIMULATIONS =====

# Initialize Storage
score_sims = [[], [], [], [], [], [], [], [], [], []]
board_sims = [[], [], [], [], [], [], [], [], [], []]

# Run Simulations
for seed in range(n_sims):
    # Episode 0
    if __name__ == "__main__":
        env = Board_3D(rows=3, cols=3, pipes=3)
        input_shape = (env.rows, env.cols, env.pipes)
        action_size = 6
        random.seed(seed)
        np.random.seed(seed)

        agent = None
        start_episode = 0
        scores = []

        agent, scores, max_tiles, accum_scores0, final_boards0 = train_agent(env, episodes=2, agent=agent, start_episode=start_episode, sim=True)
        score_sims[0].append(accum_scores0[1])
        board_sims[0].append(final_boards0[1])

    # To load checkpoint and run the remaining episode simulations
    for i, ep in enumerate(eps):
        if __name__ == "__main__":
            env = Board_3D(rows=3, cols=3, pipes=3)
            input_shape = (env.rows, env.cols, env.pipes)
            action_size = 6
            random.seed(seed)
            np.random.seed(seed)

            # Input the correct name of checkpoint file
            checkpoint_path = 'checkpoint_episode_' + str(ep) + '.pth'

            # Load parameters
            agent = DQNAgent(input_shape, action_size)
            start_episode, scores = load_checkpoint(agent, checkpoint_path, device)
            print(f"Loaded checkpoint from episode {start_episode}")

            # Continue training
            agent, scores, max_tiles, accum_score, final_board = train_agent(env, episodes=1, agent=agent, start_episode=start_episode, sim=True)
            score_sims[i + 1].append(accum_score[0])
            board_sims[i + 1].append(final_board[0])


### ===== DATA SAVING & STATISTICS =====

if save_data:
    np.save("train_sims_SCORES.npy", np.array(score_sims, dtype=object))
    np.save("train_sims_FINAL_TILES.npy", np.array(board_sims, dtype=object))

print(
    pd.DataFrame(
    {
        'min': [int(np.min([j[-1] for j in sim])) for sim in score_sims],
        'Q1': [int(sim[np.argsort([j[-1] for j in sim])[int(0.25 * len(score_sims[0])) - 1]][-1]) for sim in score_sims],
        'median': [int(sim[np.argsort([j[-1] for j in sim])[int(0.5 * len(score_sims[0])) - 1]][-1]) for sim in score_sims],
        'mean': [np.mean([j[-1] for j in sim]) for sim in score_sims],
        'Q3': [int(sim[np.argsort([j[-1] for j in sim])[int(0.75 * len(score_sims[0])) - 1]][-1]) for sim in score_sims],
        'max': [int(np.max([j[-1] for j in sim])) for sim in score_sims]
    })
)


### ===== PLOTTING =====

# 1) ------

# Plotting BEST accumulated game progress
plt.figure(figsize=(8, 6))

# Plot all game simulations
for j, sim in enumerate(score_sims):
    for game in sim: plt.plot(game, color=plt.cm.tab20((j * 2 + 1)/20), alpha=0.04)
# Plot best scoring game
for k, ep_score in enumerate(score_sims):
    plt.plot(ep_score[np.argmax([j[-1] for j in ep_score])], color=plt.cm.tab20((k * 2)/20), label=ep_labs[k])

plt.title('Best Training Game Progression, by Episode')
plt.xlabel('Number of Moves')
plt.ylabel('Accumulated Reward')
plt.legend(loc='lower right')
plt.tight_layout()
if save_plots: plt.savefig('training_progress_best.png')
plt.show()


# 2) ------

# Plot desired quantile
quantile = 75
plt.figure(figsize=(8, 6))

# Plot all game simulations
for j, sim in enumerate(score_sims):
    for game in sim: plt.plot(game, color=plt.cm.tab20((j * 2 + 1)/20), alpha=0.04)
# Plot desired game
for k, sim in enumerate(score_sims):
    plt.plot(sim[np.argsort([j[-1] for j in sim])[int(quantile * len(score_sims[0])/100) - 1]], label=ep_labs[k])

plt.title('Episode Game Progression, ' + str(quantile) + 'th Quantile')
plt.xlabel('Number of Moves')
plt.ylabel('Accumulated Reward')
plt.legend(loc='lower right')
plt.tight_layout()
if save_plots: plt.savefig('training_progress_' + str(quantile) + 'qntl.png')
plt.show()


# 3) ------

# Box plots of final scores
plt.figure(figsize=(8, 6))
plt.boxplot([[v[-1] for v in s] for s in score_sims], tick_labels=ep_labs)
plt.title('Training Final Scores')
plt.tight_layout()
if save_plots: plt.savefig('training_score_boxplot.png')
plt.show()