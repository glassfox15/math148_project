import numpy as np

class Board_3D:
    def __init__(self, rows=3, cols=3, pipes=3, prob_2=0.9, finish_value=2048):
        self.rows = rows
        self.cols = cols
        self.pipes = pipes
        self.prob_2 = prob_2
        self.finish_value = finish_value
        self.directions = ['Q', 'W', 'E', 'A', 'S', 'D']  # Action mapping
        self.reset()

    def reset(self):
        self.board = np.zeros((self.rows, self.cols, self.pipes), dtype=np.int64)
        self.add_2_4()
        self.game_over = False
        self.score = 0
        return self.board.copy()

    def add_2_4(self):
        open_pos = np.argwhere(self.board == 0)
        if open_pos.size > 0:
            new_pos = open_pos[np.random.choice(len(open_pos))]
            self.board[tuple(new_pos)] = 2 if np.random.random() < self.prob_2 else 4

    def tiles_left(self, column):
        non_zero = column[column != 0]
        padded = np.concatenate([[0], non_zero])
        score = 0

        for i in range(1, len(padded)):
            if padded[i] == padded[i-1]:
                padded[i-1] *= 2
                score += padded[i-1]
                padded[i] = 0

        non_zero = padded[padded != 0]
        new_col = np.concatenate([non_zero, np.zeros(len(column) - len(non_zero))])
        return new_col, score

    def action(self, direction):
        d1, ax1, d2, ax2 = 0, (0, 1), 0, (0, 1)  # Default values
        dir_upper = direction.upper()

        if 'A' in dir_upper: d1, ax1, d2, ax2 = 0, (1, 2), 0, (2, 1)
        if 'S' in dir_upper: d1, ax1, d2, ax2 = 1, (2, 1), 0, (2, 1)
        if 'D' in dir_upper: d1, ax1, d2, ax2 = 2, (1, 2), 0, (2, 1)
        if 'W' in dir_upper: d1, ax1, d2, ax2 = 1, (1, 2), 0, (2, 1)
        if 'Q' in dir_upper: d1, ax1, d2, ax2 = 1, (1, 0), 1, (1, 2)
        if 'E' in dir_upper: d1, ax1, d2, ax2 = 1, (0, 1), 1, (1, 2)

        rotated_board = np.rot90(np.rot90(self.board, d1, axes=ax1), d2, axes=ax2)
        rotated_reshaped = rotated_board.reshape(-1, rotated_board.shape[2])

        total_score = 0
        new_columns = []

        for col in rotated_reshaped:
            processed_col, score = self.tiles_left(col)
            new_columns.append(processed_col)
            total_score += score

        new_board = np.array(new_columns).reshape(rotated_board.shape)
        restored = np.rot90(np.rot90(new_board, -d2, axes=ax2), -d1, axes=ax1)

        return restored, total_score

    def check_game_over(self):
        if np.any(self.board >= self.finish_value):
            return True
        if (self.board == 0).any():
            return False
        for i in range(self.rows):
            for j in range(self.cols):
                for k in range(self.pipes):
                    current = self.board[i,j,k]
                    if (i < self.rows-1 and current == self.board[i+1,j,k]) or \
                       (j < self.cols-1 and current == self.board[i,j+1,k]) or \
                       (k < self.pipes-1 and current == self.board[i,j,k+1]):
                        return False
        return True

    def step(self, action):
        direction = self.directions[action]
        old_board = self.board.view()  # use view instead of copy to improve performence
        new_board, delta_score = self.action(direction)
        moved = not np.array_equal(old_board, new_board)

        reward = 0
        if moved:
            self.board = new_board
            self.score += delta_score
            reward += delta_score

            max_tile = np.max(new_board)
            if max_tile > np.max(old_board):
                reward += np.log2(max_tile) * 10

            self.add_2_4()
        else:
            reward -= 1

        done = self.check_game_over()
        return self.board.copy(), reward, done, {"valid_move": moved}

    def get_valid_actions(self):
      valid_actions = []
      current_board = self.board.copy()
      for action in range(6):  # 6 actions: Q, W, E, A, S, D
        direction = self.directions[action]
        new_board, _ = self.action(direction)
        if not np.array_equal(current_board, new_board):
            valid_actions.append(action)
      return valid_actions
    
