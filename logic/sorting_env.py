import gym
import numpy as np
from gym import spaces

class SortingEnv(gym.Env):
    def __init__(self, list_size=10):
        super().__init__()
        self.list_size = list_size
        self.action_space = spaces.Discrete((list_size * (list_size - 1)) // 2)  # n*(n-1)/2 swaps
        self.observation_space = spaces.Box(low=0, high=100, shape=(list_size,), dtype=np.int32)
        self.swap_pairs = [(i, j) for i in range(list_size) for j in range(i+1, list_size)]
        self.reset()

    def count_inversions(self, arr):
        inv_count = 0
        for i in range(len(arr)):
            for j in range(i+1, len(arr)):
                if arr[i] > arr[j]:
                    inv_count += 1
        return inv_count

    def reset(self):
        self.state = np.random.randint(0, 100, self.list_size)
        self.steps = 0
        self.prev_inversions = self.count_inversions(self.state)
        return self.state

    def step(self, action):
        i, j = self.swap_pairs[action]
        reward = -1  # Cost for comparison
        swapped = False
        if self.state[i] > self.state[j]:
            self.state[i], self.state[j] = self.state[j], self.state[i]
            swapped = True
        current_inversions = self.count_inversions(self.state)
        if swapped and current_inversions < self.prev_inversions:
            reward += 1  # Reward for reducing inversions
        self.prev_inversions = current_inversions
        self.steps += 1
        done = np.all(self.state[:-1] <= self.state[1:]) or self.steps > self.list_size * 20
        if done and np.all(self.state[:-1] <= self.state[1:]):
            reward += 100  # Reward for sorted list
        return self.state, reward, done, {"swap": [i, j], "swapped": swapped}

    def render(self):
        return f"State: {self.state.tolist()}"