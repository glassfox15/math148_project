from game_implementation import *
from model_construction import *
import numpy as np
import torch
import time

"""
This file contains the function to train the model.
Note that the hyperparameters were defined in `model_construction.py`.

Functions
    load_checkpoint:    loads saved checkpoints from model training
    train_agent:        trains the AI agent, starting from checkpoint if specified
"""

device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # configure device

# Load model checkpoint
def load_checkpoint(agent, path, device=device):
    checkpoint = torch.load(path, map_location=device)
    agent.policy_net.load_state_dict(checkpoint['policy_net_state_dict'])
    agent.target_net.load_state_dict(checkpoint['target_net_state_dict'])
    agent.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    agent.epsilon = checkpoint['epsilon']

    return checkpoint['episode'], checkpoint['scores']

# Training function
def train_agent(env, episodes=100, agent=None, start_episode=0, sim=False):
    """
    Function to test the trained agent by playing the game.

    Args:
        env: The 3D 2048 game environment (Board_3D instance).
        episodes: Number of episodes to train.
        agent: The DQNAgent instance.
        start_episode: The pisode to start from.
        sim: Whether the training is simulation (True) or actually training (False)

    Returns:
        tuple containing the trained agent and scores, modified for simulations
    """

    input_shape = (env.rows, env.cols, env.pipes)
    action_size = 6  # Q, W, E, A, S, D

    # Create a new agent only if one is not provided
    if agent is None:
        agent = DQNAgent(input_shape, action_size)

    scores = []
    largest_tiles = []
    max_score = 0
    max_tile = 0
    start_time = time.time()

    # FOR SIMULATIONS (denoted with #####)
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

            episode_rewards = [] #####

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

                episode_rewards.append(score) #####

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
            largest_tiles.append(np.max(state))
            max_score = max(max_score, score)
            max_tile = max(max_tile, np.max(state))

            episode_time = time.time() - episode_start_time
            avg_score = np.mean(scores[-50:]) if episode >= 50 else np.mean(scores)
            steps_per_second = steps / episode_time

            # RECORD FOR SIMULATIONS
            accumulated_scores.append(episode_rewards)      #####
            final_boards.append(list(state.reshape(-1)))    #####

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

            # Checkpoint saving every 10 episodes (NOT FOR SIMULATIONS)
            if episode % 10 == 0 and not sim:
                checkpoint = {
                    'episode': episode,
                    'policy_net_state_dict': agent.policy_net.state_dict(),
                    'target_net_state_dict': agent.target_net.state_dict(),
                    'optimizer_state_dict': agent.optimizer.state_dict(),
                    'epsilon': agent.epsilon,
                    'scores': scores
                }
                torch.save(checkpoint, f'checkpoint_episode_{episode}.pth')
                print(f"  Saved checkpoint to checkpoint_episode_{episode}.pth")

            # GPU memory cleanup
            if episode % 10 == 0 and hasattr(torch.cuda, 'empty_cache'):
                torch.cuda.empty_cache()

    except KeyboardInterrupt:
        print("\nTraining interrupted by user")

    finally:
        total_time = time.time() - start_time
        print(f"\nTraining completed in {total_time:.1f} seconds")
        if hasattr(torch.cuda, 'empty_cache'):
            torch.cuda.empty_cache()

    if sim: return agent, scores, largest_tiles, accumulated_scores, final_boards
    
    return agent, scores, largest_tiles