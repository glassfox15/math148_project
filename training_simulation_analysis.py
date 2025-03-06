from game_implementation import *
from model_construction import *
import matplotlib.pyplot as plt
import numpy as np
import torch
import time

"""
This code simulates training runs of the model at different episodes.
The purpose is to understand possible and realized training behavior.

Requires the following PRE-TRAINED model checkpoints:
    checkpoint_episode_50.pth
    checkpoint_episode_100.pth
    checkpoint_episode_150.pth
    checkpoint_episode_170.pth
    checkpoint_episode_200.pth
    checkpoint_episode_250.pth
    checkpoint_episode_300.pth
    checkpoint_episode_340.pth
    checkpoint_episode_400.pth

Note that this code runs the training iteration WITHOUT updating the parameters or
updating the checkpoint. The final result is a set of plots dsplaying the simulations.
"""

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Prepare Simulations
n_sims = 1000
eps = [50, 100, 150, 170, 200, 250, 300, 350, 400] # episodes to load
ep_labs = ['ep 0', *['ep ' + str(e) for e in eps]] # episode labels


### ===== MODEL TRAINING =====
# This version is modified for simulations

def train_agent(env, episodes=100, agent=None, start_episode=0):
    input_shape = (env.rows, env.cols, env.pipes)
    action_size = 6  # Q, W, E, A, S, D

    # Create a new agent only if one is not provided
    if agent is None:
        agent = DQNAgent(input_shape, action_size)

    scores = []
    max_score = 0
    max_tile = 0
    start_time = time.time()

    ### MODIFIED FROM ORIGINAL
    accumulated_scores = []
    final_boards = []

    # Performance Monitoring
    times = {
        'state_processing': 0,
        'action_selection': 0,
        'env_step': 0,
        'memory_push': 0,
        'optimization': 0
    }

    print("Starting training...")

    try:
        for episode in range(start_episode, start_episode + episodes):
            episode_start_time = time.time()
            state = env.reset()
            score = 0
            done = False
            steps = 0
            valid_moves = 0
            invalid_moves = 0

            episode_rewards = [] ### MODIFIED FROM ORIGINAL

            print(f"\nStarting Episode {episode}")

            while not done:
                steps += 1

                t0 = time.time()
                processed_state = agent.preprocess_single_state(state)
                times['state_processing'] += time.time() - t0

                t0 = time.time()
                action = agent.select_action(processed_state)
                times['action_selection'] += time.time() - t0

                t0 = time.time()
                next_state, reward, done, info = env.step(action.item())
                times['env_step'] += time.time() - t0

                if info['valid_move']:
                    valid_moves += 1
                else:
                    invalid_moves += 1

                t0 = time.time()
                agent.memory.push(
                    state,
                    action.item(),
                    reward,
                    next_state if not done else None,
                    done
                )
                times['memory_push'] += time.time() - t0

                state = next_state
                score += reward

                episode_rewards.append(score)  ### MODIFIED FROM ORIGINAL

                t0 = time.time()
                agent.optimize_model()
                times['optimization'] += time.time() - t0

                # Print training status every 100 steps
                if steps % 100 == 0:
                    current_max_tile = np.max(state)
                    print(f"  Step {steps}: Score={score:.1f}, MaxTile={current_max_tile}, "
                          f"Valid/Invalid moves={valid_moves}/{invalid_moves}")

                # Safety check: terminate episode if too many steps
                if steps > 5000:
                    print("  Reached step limit - ending episode")
                    done = True

            # Update epsilon and target network after each episode
            agent.update_epsilon()
            if episode % TARGET_UPDATE == 0:
                agent.update_target_net()

            # Record the scores and update statistics
            scores.append(score)
            max_score = max(max_score, score)
            max_tile = max(max_tile, np.max(state))

            # MODIFIED FROM ORIGINAL ###
            accumulated_scores.append(episode_rewards)
            final_boards.append(list(state.reshape(-1)))

            episode_time = time.time() - episode_start_time
            avg_score = np.mean(scores[-50:]) if episode >= 50 else np.mean(scores)
            steps_per_second = steps / episode_time

            # Episode summary
            print(f"\nEpisode {episode} Summary:")
            print(f"  Score: {score:.1f} | Avg: {avg_score:.1f} | Max: {max_score:.1f}")
            print(f"  Max Tile: {max_tile} | Steps: {steps} | Time: {episode_time:.1f}s")
            print(f"  Steps/second: {steps_per_second:.1f}")
            print(f"  Valid/Invalid moves: {valid_moves}/{invalid_moves}")
            print(f"  Epsilon: {agent.epsilon:.3f}")

            # Print performance statistics every 10 episodes
            if episode % 10 == 0:
                print("\nPerformance Statistics:")
                total_time = sum(times.values())
                for key, value in times.items():
                    percentage = (value / total_time * 100) if total_time > 0 else 0
                    print(f"  {key}: {value:.2f}s ({percentage:.1f}%)")
                times = {k: 0 for k in times}

            ### CHECKPOINT UPDATING REMOVED FOR SIMULATIONS

            # GPU memory cleanup
            if episode % 10 == 0 and hasattr(torch.cuda, 'empty_cache'):
                torch.cuda.empty_cache()

    finally:
        total_time = time.time() - start_time
        print(f"\nTraining completed in {total_time:.1f} seconds")
        if hasattr(torch.cuda, 'empty_cache'):
            torch.cuda.empty_cache()

    return agent, scores, accumulated_scores, final_boards # OUTPUTS MODIFIED FROM ORIGINAL


# Load Existing Model from Path
def load_checkpoint(agent, path):
    checkpoint = torch.load(path, map_location=torch.device('cpu'))
    agent.policy_net.load_state_dict(checkpoint['policy_net_state_dict'])
    agent.target_net.load_state_dict(checkpoint['target_net_state_dict'])
    agent.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    agent.epsilon = checkpoint['epsilon']

    return checkpoint['episode'], checkpoint['scores']


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

        agent, scores, accum_scores0, final_boards0 = train_agent(env, episodes=2, agent=agent, start_episode=start_episode)
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
            start_episode, scores = load_checkpoint(agent, checkpoint_path)
            print(f"Loaded checkpoint from episode {start_episode}")

            # Continue training
            agent, scores, accum_score, final_board = train_agent(env, episodes=1, agent=agent, start_episode=start_episode)
            score_sims[i + 1].append(accum_score[0])
            board_sims[i + 1].append(final_board[0])


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

plt.title('Best Game Progression, by Episode')
plt.xlabel('Number of Moves')
plt.ylabel('Accumulated Reward')
plt.legend(loc='lower right')
plt.tight_layout()


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

plt.title('Episode Game Progression, ' + str(quantile) + '-th Quantile')
plt.xlabel('Number of Moves')
plt.ylabel('Accumulated Reward')
plt.legend(loc='lower right')
plt.tight_layout()


