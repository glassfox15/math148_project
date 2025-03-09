from game_implementation import Board_3D
from model_construction import *
from testing import ai_play
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import torch
import time

"""
This code simulates testing runs of the model at different episodes.
The purpose is to understand possible agent game play outcomes.

Requires the following PRE-TRAINED model checkpoints:
    checkpoint_episode_0.pth <
    checkpoint_episode_50.pth
    checkpoint_episode_100.pth
    checkpoint_episode_150.pth
    checkpoint_episode_190.pth
    checkpoint_episode_200.pth
    checkpoint_episode_250.pth
    checkpoint_episode_300.pth
    checkpoint_episode_340.pth
    checkpoint_episode_400.pth

A maximum of 10 checkpoints can be visualized at once.
"""

save_data = False        # whether you want to store the data
save_plots = False       # whether you want to store the plots

# Prepare Simulations
n_sims = 1000
eps = [0, 50, 100, 150, 190, 200, 250, 300, 350, 400] # episodes to load
ep_labs = ['ep ' + str(e) for e in eps] # episode labels

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


### ===== SIMULATIONS =====

# Initialize Storage
test_score_sims = []
test_board_sims = []
test_action_sims = []

# Run Simulations
for i, ep in enumerate(eps):
    if __name__ == "__main__":
        random.seed(90095)
        np.random.seed(90095)
        
        # Define parameters
        input_shape = (3, 3, 3)  # rows, cols, pipes
        action_size = 6  # Number of possible actions (Q, W, E, A, S, D)
        checkpoint_path = 'checkpoint_episode_' + str(ep) + '.pth'

        # Initialize the agent
        trained_agent = DQNAgent(input_shape, action_size)

        # Load the trained model from the checkpoint
        try:
            checkpoint = torch.load(checkpoint_path, weights_only=False, map_location=device)
            trained_agent.policy_net.load_state_dict(checkpoint['policy_net_state_dict'])
            trained_agent.policy_net.eval()  # Set to evaluation mode
            print(f"Successfully loaded model from {checkpoint_path}")
        except FileNotFoundError:
            print(f"Error: Checkpoint file '{checkpoint_path}' not found. Please ensure training has completed and the file exists.")
            exit(1)

        # Create the environment
        env = Board_3D(rows=3, cols=3, pipes=3)

        # Do 1000 iterations using this model
        ai_results = ai_play(env, trained_agent, num_episodes=1000, render=False, sim=True)
        print('Episode ' + str(i) + ' done')

    test_score_sims.append([a['accum_score'] for a in ai_results])
    test_board_sims.append([a['max_tile'] for a in ai_results])
    test_action_sims.append([a['moves_tracking'] for a in ai_results])


### ===== DATA SAVING & STATISTICS =====

if save_data:
    np.save("test_sims_SCORES.npy", np.array(test_score_sims, dtype=object))
    np.save("test_sims_FINAL_TILES.npy", np.array(test_board_sims, dtype=object))
    np.save("test_sims_ACTIONS.npy", np.array(test_action_sims, dtype=object))

print(
    pd.DataFrame(
    {
        'min': [int(np.min([j[-1] for j in sim])) for sim in test_score_sims],
        'Q1': [int(sim[np.argsort([j[-1] for j in sim])[int(0.25 * len(test_score_sims[0])) - 1]][-1]) for sim in test_score_sims],
        'median': [int(sim[np.argsort([j[-1] for j in sim])[int(0.5 * len(test_score_sims[0])) - 1]][-1]) for sim in test_score_sims],
        'mean': [np.mean([j[-1] for j in sim]) for sim in test_score_sims],
        'Q3': [int(sim[np.argsort([j[-1] for j in sim])[int(0.75 * len(test_score_sims[0])) - 1]][-1]) for sim in test_score_sims],
        'max': [int(np.max([j[-1] for j in sim])) for sim in test_score_sims]
    })
)


### ===== PLOTTING =====

# 1) ------

plt.figure(figsize=(8, 6))

# Plot all game simulations
for j, sim in enumerate(test_score_sims):
    for game in sim: plt.plot(game, color=plt.cm.tab20((j * 2 + 1)/20), alpha=0.25 ** ((j + 1)/4))

# Plot best scoring game
for k, ep_score in enumerate(test_score_sims):
    plt.plot(ep_score[np.argmax([j[-1] for j in ep_score])], color=plt.cm.tab20((k * 2)/20), label=ep_labs[k])

plt.title('Best Testing Game Progression, by Episode')
plt.xlabel('Number of Moves')
plt.ylabel('Accumulated Reward')
plt.legend(loc='lower right')
plt.tight_layout()
if save_plots: plt.savefig('testing_progress_best.png')
plt.show()


# 2) ------

# Plot desired quantile
quantile = 75
plt.figure(figsize=(8, 6))

# Plot all game simulations
for j, sim in enumerate(test_score_sims):
    for game in sim: plt.plot(game, color=plt.cm.tab20((j * 2 + 1)/20), alpha=0.25 ** ((j + 1)/4))
# Plot desired game
for k, sim in enumerate(test_score_sims):
    plt.plot(sim[np.argsort([j[-1] for j in sim])[int(quantile * n_sims / 100) - 1]], label=ep_labs[k])

plt.title('Episode Game Progression, ' + str(quantile) + 'th Quantile')
plt.xlabel('Number of Moves')
plt.ylabel('Accumulated Reward')
plt.legend(loc='lower right')
plt.tight_layout()
if save_plots: plt.savefig('testing_progress_' + str(quantile) + 'qntl.png')
plt.show()


# 3) ------

# Box plots of final scores
plt.figure(figsize=(8, 6))
plt.boxplot([[v[-1] for v in s] for s in test_score_sims], tick_labels=ep_labs)
plt.title('Testing Final Scores')
plt.tight_layout()
if save_plots: plt.savefig('testing_score_boxplot.png')
plt.show()


# 4) ------

### OPERATION DISTRIBUTIONS: Episode 190
ep190_actions = test_action_sims[[i for i in range(len(eps)) if eps[i] == 190][0]]

move_dist_arr = np.array([[r.count(move) for r in ep190_actions] for move in set(ep190_actions[0])])
move_props = move_dist_arr / move_dist_arr.sum(axis=0)

pd.DataFrame(np.flip(np.sort(move_props, axis=0),axis=0).T).plot(
    kind='bar',
    stacked=True,
    xlabel='Game',
    ylabel='Move Relative Proportion',
    width=1.0,
    title='Game Move Distributions (Ep 190)',
    figsize=(7, 5)
)
plt.gca().set_xticklabels([])
plt.gca().set_xticks([])
plt.ylim((0,1))
plt.legend(['most freq', '', '', '', '', 'least Freq'], loc='lower right')
plt.tight_layout()
if save_plots: plt.savefig('testing_action_dist_ep190.png')
plt.show()


# 5) ------

### OPERATION DISTRIBUTIONS: Episode 340
ep340_actions = test_action_sims[[i for i in range(len(eps)) if eps[i] == 340][0]]

move_dist_arr = np.array([[r.count(move) for r in ep340_actions] for move in set(ep340_actions[0])])
move_props = move_dist_arr / move_dist_arr.sum(axis=0)

pd.DataFrame(np.flip(np.sort(move_props, axis=0),axis=0).T).plot(
    kind='bar',
    stacked=True,
    xlabel='Game',
    ylabel='Move Relative Proportion',
    width=1.0,
    title='Game Move Distributions (Ep 340)',
    figsize=(7, 5)
)
plt.gca().set_xticklabels([])
plt.gca().set_xticks([])
plt.ylim((0,1))
plt.legend(['most freq', '', '', '', '', 'least Freq'], loc='lower right')
plt.tight_layout()
if save_plots: plt.savefig('testing_action_dist_ep340.png')
plt.show()


