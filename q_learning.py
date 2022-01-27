import numpy as np
from collections import defaultdict

class QLearningModel:
    def __init__(self, pipe_width, pipe_gap, player_height, player_width):
        self.eps = 0.01
        self.eps_decay_rate = 0
        self.gamma = 0.9
        self.alpha = 0.7

        self.pipe_width = pipe_width
        self.pipe_gap = pipe_gap
        self.player_height = player_height
        self.player_width = player_width

        # initial value is optimistic for more exploration at beginning 
        self.model = defaultdict(lambda: 0)

        self.games_played = 0

    def round(self, num, factor, lo=None, hi=None):
        rounded = int(factor * (num // factor))
        if lo:
            rounded = max(rounded, lo)
        if hi:
            rounded = min(rounded, hi)
        return rounded

    def get_state_action(self, state, action):
        state = [self.round(state[0], 15, hi=200), self.round(state[1], 15), self.round(state[2], 2)]
        state_action = tuple(state + [int(action)])
        return state_action

    def get_q(self, state, action):
        state_action = self.get_state_action(state, action)
        return self.model[state_action]

    def set_q(self, state, action, q):
        state_action = self.get_state_action(state, action)
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
            return int(self.get_q(state, 1) > self.get_q(state, 0))
        else:
            # exploration
            return int(np.random.uniform() > 0.5)

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
                lower_pipe_middle = lower_pipe['y'] - self.pipe_gap / 2
                y_middle = y_top + self.player_height / 2
                y_dist_to_pipe = y_middle - lower_pipe_middle # positive if pipe is high up
                state = [x_dist_to_pipe, y_dist_to_pipe, y_vel]
                return np.array(state)

    def process_snapshots(self, snapshots):
        N = len(snapshots)
        for i in range(N): 
            unprocessed_state, score, action, _ = snapshots[i]
            state = self.get_state(unprocessed_state)
            if i != N - 1:
                new_unprocessed_state, new_score, _, _ = snapshots[i + 1]
                new_state = self.get_state(new_unprocessed_state)
                reward = 10
            else:
                new_state = None
                #amount_off_center = abs(state[1]) / 512
                reward = -1000
            reward += 100 * (new_score - score)
            self.process_snapshot(state, action, reward, new_state)
        print(f'    Score: {snapshots[-1][1]}')

        # decay epsilon over time
        self.eps *= (1 - self.eps_decay_rate)

        # print status
        self.games_played += 1
        if self.games_played % 5 == 0:
            print(f'Played {self.games_played} games.')
            print(f'  Dictionary size: {len(self.model)}. Average q-value: {np.mean(list(self.model.values())):.3g}')
            print(f'  Epsilon: {self.eps:.3g}.')
            print(f'  Discount factor (gamma): {self.gamma}. Learning rate (alpha): {self.alpha}.')

if __name__ == '__main__':
    pass