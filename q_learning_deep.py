import torch
from torch import nn, optim
import torch.nn.functional as f
import numpy as np
import copy
import random

class QNetwork:
    def __init__(self, pipe_width, pipe_gap, player_height):
        self.eps = 0.05
        self.eps_decay_rate = 0.001
        self.delta = 0.9
        self.q_network = nn.Sequential(
            nn.Linear(6, 10),
            nn.ReLU(),
            nn.Linear(10, 2)
        )
        self.target_network = copy.deepcopy(self.q_network)
        self.target_network.train()

        self.target_update_freq = 20
        self.games = 0
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=0.1)

        self.replay_buffer = []
        self.batch_size = 100

        self.pipe_width = pipe_width
        self.pipe_gap = pipe_gap
        self.player_height = player_height

    def get_q_vals(self, state, eval_mode, no_grad):
        if eval_mode:
            self.q_network.eval()
        else:
            self.q_network.train()
        
        if no_grad:
            with torch.no_grad():
                q_vals = self.q_network(state)
        else:
            with torch.enable_grad():
                q_vals = self.q_network(state)
        return q_vals

    def get_action_eps_greedy(self, unprocessed_state):
        if np.random.uniform() > self.eps: 
            # exploitation
            state, _ = self.get_state(unprocessed_state)
            q_vals = self.get_q_vals(state, eval_mode=True, no_grad=True)
            return q_vals[1] > q_vals[0]
        else:
            # exploration
            return np.random.uniform() > 0.5

    def get_state(self, unprocessed_state):
        x_left, y_top, y_vel, lower_pipes = unprocessed_state
        state = [y_top, y_vel]
        pipes_left = 2
        for lower_pipe in lower_pipes:
            lower_pipe_right = lower_pipe['x'] + self.pipe_width
            if x_left < lower_pipe_right:
                state += [lower_pipe['x'], lower_pipe['y']]
                pipes_left -= 1
                if pipes_left == 0:
                    break
        while pipes_left > 0:
            state += [0, 0]
            pipes_left -= 1
        assert len(state) == 6
        # get reward
        y_pipe_center = state[3] - self.pipe_gap / 2
        y_center = y_top + self.player_height / 2
        norm_y_dist_to_pipe = abs(y_pipe_center - y_center) / 512
        return torch.Tensor(state), norm_y_dist_to_pipe

    def update_weights(self, batch):
        #state_batch = [e[0] for e in batch]
        #new_state_batch = [e[1] for e in batch]
        #action_batch = [e[2] for e in batch]
        #reward_batch = [e[3] for e in batch]

        loss = 0
        for example in batch:
            state, new_state, action, reward = example
            # get q-value for original state (gradients turned on)
            q_vals = self.get_q_vals(state, eval_mode=False, no_grad=False)
            q_val = q_vals[int(action)]
            # get target q-value (gradients turned off)
            if new_state is not None:
                with torch.no_grad():
                    new_state_q_vals = self.target_network(new_state)
                target_q_val = reward + self.delta * torch.max(new_state_q_vals)
            else:
                target_q_val = reward
            # compute loss 
            temp_diff = target_q_val - q_val
            loss += torch.square(temp_diff)
        loss /= len(batch)
        
        # update weights
        loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()
        return loss

    def process_snapshots(self, snapshots):
        # compute and store states of last game run in replay buffer
        N = len(snapshots)
        loss = 0
        total_reward = 0
        for i in range(N): 
            unprocessed_state, score, action, _ = snapshots[i]
            state, dist_to_pipe = self.get_state(unprocessed_state)
            if i != N - 1:
                new_unprocessed_state, new_score, _, _ = snapshots[i + 1]
                new_state, _ = self.get_state(new_unprocessed_state)
                reward = 0.1
            else:
                new_state = None
                reward = -1 * dist_to_pipe
            reward += 2 * (new_score - score)
            self.replay_buffer.append([state, new_state, action, reward])
            total_reward += reward
        print(f'Average reward: {total_reward / N:.3g}')

        # sample from replay buffer and update weights 
        batch = random.sample(self.replay_buffer, min(self.batch_size, len(self.replay_buffer)))
        loss = self.update_weights(batch)
        print(f'Epoch loss: {loss.item():.3g}')

        # update target network if needed
        self.games += 1
        if self.games == self.target_update_freq:
            self.target_network = copy.deepcopy(self.q_network)
            self.target_network.train()
            self.games = 0
            print('Copying Q-network to target network.')

        # decay epsilon
        self.eps *= (1 - self.eps_decay_rate)

if __name__ == '__main__':
    pass