from game_implementation import Board_3D
from model_construction import DQNAgent
from testing import ai_play
import matplotlib.pyplot as plt
#import numpy as np
import torch

# Inputs
checkpoint_path = None          # load saved agent: 'checkpoint_episode_###.pth'
games = 3                       # desired number of games to play

# Define device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if __name__ == "__main__":
    # Define parameters
    input_shape = (3, 3, 3)  # rows, cols, pipes
    action_size = 6  # Number of possible actions (Q, W, E, A, S, D)
    checkpoint_path = 'checkpoint_episode_190.pth'  # Adjust this to your last checkpoint

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

    # Run the AI for 100 episodes with visualization
    ai_results = ai_play(env, trained_agent, num_episodes=100, render=True, sim=False)