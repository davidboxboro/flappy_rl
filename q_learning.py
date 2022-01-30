import numpy as np
from collections import defaultdict
import sys
import dill

class QLearningModel:
    def __init__(self, pipe_width, pipe_gap, player_height, player_width, load=False):
        self.eps = 0.01
        self.eps_decay_rate = 0.0001
        self.eps_min = 0.0001
        self.gamma = 0.9
        self.alpha = 0.7

        self.pipe_width = pipe_width
        self.pipe_gap = pipe_gap
        self.player_height = player_height
        self.player_width = player_width

        init_val = 0
        if not load:
            # initial value is optimistic for more exploration at beginning 
            self.model = defaultdict(lambda: init_val)
        else:
            with open(f'model_params/40000.params', 'rb') as f:
                self.model = dill.loads(f.read())

        self.games_played = 0
        self.game_scores = []

    def save(self, i):
        with open(f'model_params/{i}.params', 'wb') as f:
            f.write(dill.dumps(self.model))

    def round(self, num, factor, lo=None, hi=None):
        rounded = int(factor * (num // factor))
        if lo:
            rounded = max(rounded, lo)
        if hi:
            rounded = min(rounded, hi)
        return rounded

    def get_state_action(self, state, action):
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

    def get_action(self, unprocessed_state):
        state = self.get_state(unprocessed_state)
        return int(self.get_q(state, 1) > self.get_q(state, 0)) 

    def get_action_eps_greedy(self, unprocessed_state):
        if np.random.uniform() > self.eps:
            # exploitation
            return self.get_action(unprocessed_state)
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
                state = [self.round(x_dist_to_pipe, 5), self.round(y_dist_to_pipe, 5), self.round(y_vel, 2)]
                return np.array(state)

    def process_snapshots(self, snapshots):
        N = len(snapshots)
        last_unprocessed_state, score, action, _ = snapshots[-1]
        _, y_dist_at_death, _ = self.get_state(last_unprocessed_state)
        if y_dist_at_death < 0:
            penalize_last_jump = True
        else:
            penalize_last_jump = False
        for i in range(N - 1, -1, -1):
            unprocessed_state, score, action, _ = snapshots[i]
            state = self.get_state(unprocessed_state)
            if i != N - 1:
                new_unprocessed_state, new_score, _, _ = snapshots[i + 1]
                new_state = self.get_state(new_unprocessed_state)
                reward = 1
            else:
                new_state = None
                reward = -1000
            if penalize_last_jump and action:
                reward -= 1000
                penalize_last_jump = False
            self.process_snapshot(state, action, reward, new_state)
            #print(unprocessed_state[-1])
        self.game_scores.append(snapshots[-1][1])

        # decay epsilon over time
        if self.eps > self.eps_min:
            self.eps *= (1 - self.eps_decay_rate)

        # print status
        self.games_played += 1
        if self.games_played % 200 == 0:
            print(f'Played {self.games_played} games.')
            print(f'  Dictionary size: {len(self.model)}. Average q-value: {np.mean(list(self.model.values())):.3g}')
            print(f'  Epsilon: {self.eps:.3g}.')
            print(f'  Discount factor (gamma): {self.gamma}. Learning rate (alpha): {self.alpha}.')
            print(f'  Rolling average score (last 500 trials): {np.mean(self.game_scores[-500:]):.3g}')
            print(f'  Max score: {np.max(self.game_scores)}')
            print(state[1])

if __name__ == '__main__':
    pass