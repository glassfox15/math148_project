from game_implementation import Board_3D
from model_construction import DQNAgent
import pygame
from pygame.locals import *
from OpenGL.GL import *
from OpenGL.GLU import *
import numpy as np
import torch

# Inputs
episode = 340           # specify saved model
moves_per_sec = 10      # play speed (moves per second)
games = 2               # number of games to watch

# Specify agent + device
checkpoint_path = 'checkpoint_episode_' + str(episode) + '.pth'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

## ------------------
## Visualization Part
## ------------------
def Cube(size):
    hs = size / 2.0  # half size
    vertices = [ # list of corners
        [ hs,  hs, -hs],
        [ hs, -hs, -hs],
        [-hs, -hs, -hs],
        [-hs,  hs, -hs],
        [ hs,  hs,  hs],
        [ hs, -hs,  hs],
        [-hs, -hs,  hs],
        [-hs,  hs,  hs]
    ]
    edges = ( # each tuple represents drawing lines from 2 different corners
        (0,1), (1,2), (2,3), (3,0),
        (4,5), (5,6), (6,7), (7,4),
        (0,4), (1,5), (2,6), (3,7)
    )
    surfaces = ( # each tuple represents a surface made by 4 specific corners
        (0,1,2,3),
        (3,2,6,7),
        (7,6,5,4),
        (4,5,1,0),
        (0,3,7,4),
        (1,2,6,5)
    )
    return vertices, edges, surfaces



def draw_wireframe_cube(x, y, z, size, edge_color=(1,1,1)):
    """Draw just the edges of a cube at (x,y,z)."""
    vertices, edges, surfaces = Cube(size)
    glPushMatrix()
    glTranslatef(x, y, z) # translate the cube to be drawn and position x, y, z

    # Set edge color
    glColor3fv(edge_color)

    # Draw only lines
    glBegin(GL_LINES)
    for edge in edges:
        for vertex in edge:
            glVertex3fv(vertices[vertex])
    glEnd()
    glPopMatrix()

def draw_solid_cube(x, y, z, size, fill_color=(1,1,1)):
    """Draw a filled cube with black edges at (x,y,z)."""
    vertices, edges, surfaces = Cube(size)
    glPushMatrix()
    glTranslatef(x, y, z)

    # Fill faces
    glColor3fv(fill_color)
    glBegin(GL_QUADS)
    for face in surfaces:
        for vertex in face:
            glVertex3fv(vertices[vertex])
    glEnd()

    # Draw edges in black for occupied cube, unoccupied cubes have white edges
    glColor3f(0,0,0)
    glBegin(GL_LINES)
    for edge in edges:
        for vertex in edge:
            glVertex3fv(vertices[vertex])
    glEnd()

    glPopMatrix()

def color_for_value(value):
    """Return a (r,g,b) color for each tile value."""
    colors = {
        2: (0.8, 0.8, 0.8), # Light gray
        4: (0.9, 0.9, 0.5), # Pale yellow / off‐white
        8: (0.9, 0.7, 0.5), # Light peach / pale orange
        16: (0.9, 0.5, 0.5), # Soft pinkish red / salmon
        32: (0.9, 0.3, 0.5), # More saturated pink‐red
        64: (0.9, 0.1, 0.5), # Bright magenta / hot pink
        128: (0.7, 0.2, 0.5), # Pinkish purple
        256: (0.5, 0.2, 0.5), # Medium purple / violet
        512: (0.3, 0.2, 0.5), # Darker purple
        1024: (0.1, 0.2, 0.5), # Navy‐ish blue
        2048: (0.0, 0.2, 0.5) # Deeper navy / teal‐blue
    }
    return colors.get(value, (1, 1, 1))


## ----------
## Connection
## ----------
def draw_board_3d(env):
    viewport = glGetIntegerv(GL_VIEWPORT)

    cube_size = 2
    spacing = 2

    # Loop over each cell in env.board
    for i in range(env.rows):
        for j in range(env.cols):
            for k in range(env.pipes):
                value = env.board[i, j, k]
                # Convert to x, y, z
                pos_x = k*(cube_size+spacing)
                pos_y = -j*(cube_size+spacing)
                pos_z = i*(cube_size+spacing)

                if value == 0:
                    draw_wireframe_cube(pos_x, pos_y, pos_z, cube_size, (0.7, 0.7, 0.7))
                else:
                    fill_color = color_for_value(value)
                    draw_solid_cube(pos_x, pos_y, pos_z, cube_size, fill_color)

def load_trained_agent(path, device=device):
    # Create the same input_shape
    input_shape = (3, 3, 3)  # rows, cols, pipes
    action_size = 6
    agent = DQNAgent(input_shape, action_size)

    checkpoint = torch.load(path, weights_only = False, map_location=device)
    agent.policy_net.load_state_dict(checkpoint['policy_net_state_dict'])
    agent.policy_net.eval()
    print(f"Loaded model from {checkpoint_path}")
    return agent

def visualize_agent_play(env, agent, num_episodes = 10):
    """
    Let the trained agent play multiple episodes of 3D 2048, with 3D visualization.
    Returns a list of results (one entry per episode).
    """
    pygame.init()
    display = (1080, 720)
    screen = pygame.display.set_mode(display, DOUBLEBUF | OPENGL)
    pygame.display.set_caption("3D 2048 with AI Agent")
    glClearColor(0.1, 0.1, 0.1, 1.0)

    # Basic camera setup
    gluPerspective(45, (display[0] / display[1]), 0.1, 100.0)
    glTranslatef(-4.5, 5, -30.0)
    glRotatef(30, 2, 1, 0)

    clock = pygame.time.Clock()
    running = True

    results = []  # We'll store episode stats here

    for episode in range(num_episodes):
        if not running:
            break  # If the user closed the window, stop altogether

        env.reset()
        done = False
        move_count = 0

        while running and not done:
            # Handle events (user close)
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                    break  # Exit event loop

            if not running:
                break  # user closed window, so exit the while loop

            # 1) Agent picks an action
            valid_actions = env.get_valid_actions()
            if valid_actions:
                state = env.board.copy()
                processed_state = agent.preprocess_single_state(state)

                with torch.no_grad():
                    q_values = agent.policy_net(processed_state).squeeze(0)
                best_action = max(valid_actions, key=lambda a: q_values[a].item())

                # 2) Step the environment
                new_state, reward, done, info = env.step(best_action)
                move_count += 1

            else:
                # No valid moves => game over
                done = True

            # 3) Render
            glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
            glEnable(GL_DEPTH_TEST)
            draw_board_3d(env)
            pygame.display.flip()

            clock.tick(moves_per_sec)

        # Episode ended
        max_tile = np.max(env.board)
        final_score = env.score

        # Store & print results for this episode
        results.append({
            'episode': episode + 1,
            'score': final_score,
            'max_tile': max_tile,
            'moves': move_count
        })
        print(f"\nEpisode {episode+1} finished!")
        print(f"Final Score: {final_score}")
        print(f"Max Tile: {max_tile}")
        print(f"Total Moves: {move_count}")
        print("="*50)

    pygame.quit()
    return results


def main():
    # 1) Create environment
    env = Board_3D(rows=3, cols=3, pipes=3)

    # 2) Load trained agent
    agent = load_trained_agent(path=checkpoint_path)

    # 3) Visualize for specified number of episodes
    results = visualize_agent_play(env, agent, num_episodes = games)

    # Print overall stats:
    avg_score = np.mean([r['score'] for r in results])
    best_score = max([r['score'] for r in results])
    print(f"\nRan {len(results)} episodes. Avg Score = {avg_score:.1f}, Best Score = {best_score}")

if __name__ == "__main__":
    main()




