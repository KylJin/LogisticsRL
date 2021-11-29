class RuleAgent(object):
    def __init__(self, key, connections, n_actions, low, high):
        self.key = key
        self.connections = connections
        self.n_actions = n_actions
        self.action_low = low
        self.action_high = high

    def choose_action(self, observation):
        storage = observation[self.key]

        action = []
        for i, nbr in enumerate(self.connections):
            high = self.action_high[i]
            avg = (storage + observation[nbr]) / 2
            if storage > avg:
                act = storage - avg
                if act <= high:
                    action.append(act)
                else:
                    action.append(high)
            else:
                action.append(0)

        return action

    """
    def choose_action(self, observation):
        storage = observation[self.key]

        avg_storage = observation[self.key]
        nbr_storages = []
        for nbr in self.connections:
            avg_storage += observation[nbr]
            nbr_storages.append(observation[nbr])
        avg_storage /= (self.n_actions + 1)

        if storage <= avg_storage:
            action = [0] * self.n_actions
            return action

        total_lack_storage = 0
        lack_storages = []
        for nbr in self.connections:
            lack = avg_storage - observation[nbr]
            if lack > 0:
                lack_storages.append(lack)
                total_lack_storage += lack
            else:
                lack_storages.append(0)

        over_storage = storage - avg_storage
        action = []
        for i in range(self.n_actions):
            high = self.action_high[i]
            act = over_storage * lack_storages[i] / total_lack_storage
            if act <= high:
                action.append(act)
            else:
                action.append(high)

        return action
    """
