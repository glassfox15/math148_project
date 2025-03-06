# Training a Computer to Play 2048 in 3D

<img src="game_picture_1.png" alt="game visual" width="400"/>

By Tianlin Yue, Yizhuo Chang, Stephanie Su, Zizhan Wei, Jacob Titcomb, Yinqi Yao, and Zhekai Zheng

This respository is a part of a capstone project for MATH 148 at UCLA entitled "Developing an AI Agent for 3D 2048 Using Deep Q-Learning."

Feel free to try for yourself!

## *Overview*

Our project studies a 3-dimensional variation on the game 2048, originally designed by [Gabriele Cirulli](https://github.com/gabrielecirulli/2048). We construct the game environment and train an AI agent to play the game for a $3\times 3\times 3$ board. Employing a Dueling Deep Q-Learning model, the code in this repository documents our development process: neural network design to model training and eventually model testing.

Unfortunately, the checkpoint files are too large to upload to GitHub, but they can be accessed through [this Google drive](https://drive.google.com/drive/folders/1L27GUpmwOVPkFXj1C2p-MhDYwcXUKvj-?usp=sharing).


## *Repository Summary*

1. `game_implementation.py`implements the 3-dimensional game, playable by user or computer.

2. Running `playable_game.py` allows the user to play 2048, with specifiable parameters such as the dimensions of the game environment and the desired finishing tile value.

3. `random_baseline.py` creates a baseline agent to compare model performance. This implementation plays 10,000 games using purely random inputs and produces summarizing graphs upon completion.

4. Getting to the heart of the project, `model_construction.py` constructs the dueling deep q-learning model. The neural network is built primarily using `PyTorch`.

5. `model_train.py` runs training episodes for the model.

6. Finally, `model_test.py` lets the AI agent play the game for itself! You can then visualize the gameplay using our implementation with `pygame` and `OpenGL` in file `visualize_game.py`.


As part of our analysis, we ran simulations of both the training and testing phases. To visualize simulated training iterations, `training_simulation_analysis.py` runs simulation batches and produces plots and statistics at the end. Similarly, `testing_simulation_analysis.py` produces plots and statistics for simulations of the testing phase, letting the models play simulated games. Just make sure you have the proper checkpoints loaded!
