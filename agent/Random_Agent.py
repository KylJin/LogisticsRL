import random


class RandomAgent(object):
    def __init__(self, key, connections, n_actions, low, high):
        self.key = key
        self.connections = connections
        self.n_actions = n_actions
        self.action_low = low
        self.action_high = high

    def choose_action(self, observation):
        action = []
        for i in range(self.n_actions):
            low = self.action_low[i]
            high = self.action_high[i]
            act = random.randint(low, high)
            action.append(act)

        return action
