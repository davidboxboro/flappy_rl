import numpy as np
from collections import defaultdict

class QLearningModel:
    def __init__(self, pipe_width, pipe_gap, player_height, player_width):
        self.eps = 0.05
        self.eps_decay_rate = 0.01
        self.gamma = 0.9
        self.alpha = 0.5
        self.k = 10

        self.pipe_width = pipe_width
        self.pipe_gap = pipe_gap
        self.player_height = player_height
        self.player_width = player_width

        self.model = defaultdict(lambda: -1) # self.model[state, action] = q_value

        self.games_played = 0

    def get_q(self, state, action):
        state = (self.k * np.round(state / self.k)).astype(int)
        state_action = np.append(state, int(action))
        state_action = tuple(e for e in state_action)
        return self.model[state_action]

    def set_q(self, state, action, q):
        state = (self.k * np.round(state / self.k)).astype(int)
        state_action = np.append(state, int(action))
        state_action = tuple(e for e in state_action)
        self.model[state_action] = q

    def process_snapshot(self, state, action, reward, new_state):
        if new_state is not None:
            max_q_new_state = max(self.get_q(new_state, 0), self.get_q(new_state, 1))
            q_target = reward + self.gamma * max_q_new_state
        else:
            q_target = reward
        q_new = (1 - self.alpha) * self.get_q(state, action) + self.alpha * q_target
        self.set_q(state, action, q_new)

    def get_action_eps_greedy(self, unprocessed_state):
        if np.random.uniform() > self.eps: 
            # exploitation
            state = self.get_state(unprocessed_state)
            return self.get_q(state, 1) > self.get_q(state, 0)
        else:
            # exploration
            return np.random.uniform() > 0.5

    def get_state(self, unprocessed_state):
        '''
        Returns state as a numpy array.
        '''
        x_left, y_top, y_vel, lower_pipes = unprocessed_state
        x_right = x_left + self.player_width
        for lower_pipe in lower_pipes:
            lower_pipe_right = lower_pipe['x'] + self.pipe_width
            x_dist_to_pipe = lower_pipe_right - x_right
            if x_dist_to_pipe > 0:
                lower_pipe_top = lower_pipe['y']
                y_bottom = y_top + self.player_height
                y_dist_to_pipe = y_bottom - lower_pipe_top # positive if pipe is high up
                state = [x_dist_to_pipe, y_dist_to_pipe]
                return np.array(state)

    def process_snapshots(self, snapshots):
        N = len(snapshots)
        for i in range(N): 
            unprocessed_state, score, action, _ = snapshots[i]
            state = self.get_state(unprocessed_state)
            if i != N - 1:
                new_unprocessed_state, new_score, _, _ = snapshots[i + 1]
                new_state = self.get_state(new_unprocessed_state)
                reward = 0.01
            else:
                new_state = None
                reward = -1
            reward += (new_score - score)
            self.process_snapshot(state, action, reward, new_state)

        # decay epsilon over time
        self.eps *= (1 - self.eps_decay_rate)

        # print status
        self.games_played += 1
        if self.games_played % 20 == 0:
            print(f'Played {self.games_played} games.')
            print(f'  Dictionary size: {len(self.model)}. Epsilon: {self.eps}.')
            print(f'  Discount factor (gamma): {self.gamma}. Learning rate (alpha): {self.alpha}.')

if __name__ == '__main__':
    pass