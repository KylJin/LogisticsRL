class RuleAgent(object):
    def __init__(self, key, n_actions, low, high):
        self.key = key
        self.n_actions = n_actions
        self.action_low = low
        self.action_high = high

    def choose_action(self, observation):
        obs = observation['obs']
        connections = observation['connected_player_index']

        storage = obs[self.key]
        action = []
        for i, nbr in enumerate(connections):
            high = self.action_high[i]
            avg = (storage + obs[nbr]) / 2
            if storage > avg:
                act = storage - avg
                if act <= high:
                    action.append(act)
                else:
                    action.append(high)
            else:
                action.append(0)

        return action
