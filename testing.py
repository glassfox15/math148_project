from game_implementation import Board_3D
from model_construction import *
import numpy as np
import torch

def ai_play(env, agent, num_episodes=3, render=True, sim=False):
    """
    Function to test the trained agent by playing the game.

    Args:
        env: The 3D 2048 game environment (Board_3D instance).
        agent: The trained DQNAgent instance.
        num_episodes: Number of episodes to play (default: 3).
        render: Whether to print the board and move details (default: True).
        sim: Whether the games are being used for simulations or not.

    Returns:
        List of dictionaries with results for each episode.
    """
    results = []

    for episode in range(num_episodes):
        state = env.reset()
        done = False
        total_reward = 0
        max_tile = 0
        move_count = 0

        # FOR SIMULATIONS (denoted with #####)
        game_score_tracking = []
        game_move_tracking = []

        while not done:
            # Preprocess the current state
            processed_state = agent.preprocess_single_state(state)

            # Get valid actions from the environment
            valid_actions = env.get_valid_actions()

            # Select the best action based on the policy network
            if valid_actions:
                with torch.no_grad():
                    q_values = agent.policy_net(processed_state).squeeze(0)  # Shape: [6]
                    valid_q_values = q_values[valid_actions]  # Q-values for valid actions
                    action = valid_actions[torch.argmax(valid_q_values).item()]
            else:
                # No valid actions indicate the game is over
                done = True
                break

            # Take a step in the environment
            next_state, reward, done, info = env.step(action)

            # Update metrics
            total_reward += reward
            current_max = np.max(next_state)
            max_tile = max(max_tile, current_max)
            move_count += 1

            # Track score and actions
            game_score_tracking.append(total_reward) #####
            game_move_tracking.append(env.directions[action]) #####

            # Render the game state if enabled
            if render:
                print(f"Move {move_count}:")
                print(env.board)
                print(f"Action: {env.directions[action]} | Reward: {reward}")
                print(f"Valid Move: {info['valid_move']}")
                print("-" * 30)

            state = next_state

        # Store episode results
        if sim:
            results.append({
                'episode': episode + 1,
                'score': total_reward,
                'max_tile': max_tile,
                'moves': move_count,
                'accum_score': game_score_tracking, #####
                'moves_tracking': game_move_tracking #####
            })
        else:
            results.append({
                'episode': episode + 1,
                'score': total_reward,
                'max_tile': max_tile,
                'moves': move_count
            })
        

        # Print episode summary
        print(f"Episode {episode + 1} completed!")
        print(f"Final Score: {total_reward}")
        print(f"Max Tile: {max_tile}")
        print(f"Total Moves: {move_count}")
        print("=" * 50)

    # Print overall summary
    print("\n=== AI Performance Summary ===")
    print(f"Total Episodes: {num_episodes}")
    print(f"Average Score: {np.mean([r['score'] for r in results]):.1f}")
    print(f"Best Score: {max([r['score'] for r in results])}")
    print(f"Highest Tile Achieved: {max([r['max_tile'] for r in results])}")

    return results