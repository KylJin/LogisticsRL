def my_controller(observation, action_space, is_act_continuous=False):
    obs = observation['obs']
    connections = observation['connected_player_index']
    storage = obs[observation['controlled_player_index']]

    action = []
    for i, nbr in enumerate(connections):
        high = action_space[0].high[i]
        avg = (storage + obs[nbr]) / 2
        if storage > avg:
            act = storage - avg
            if act <= high:
                action.append(act)
            else:
                action.append(high)
        else:
            action.append(0)

    return [action]
