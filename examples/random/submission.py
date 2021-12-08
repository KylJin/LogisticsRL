def my_controller(observation, action_space, is_act_continuous=False):
    agent_action = []
    for i in range(len(action_space)):
        action_ = action_space[i].sample()
        agent_action.append(action_)
    return agent_action
