import core


def value_iteration(mdp, epsilon=0.1):
    value_function = core.get_initial_value_function(mdp)

    while True:
        new_value_function = value_function.copy()
        delta = 0

        for state in mdp.states:
            state_key = core.key(state)

            current_reward = mdp.get_reward(state)
            future_reward = max(core.get_action_values(mdp, value_function, state))
            new_value_function[state_key] = current_reward + mdp.gamma * future_reward

            delta = max(delta, abs(new_value_function[state_key] - value_function[state_key]))

        value_function = new_value_function

        if delta < epsilon:
            return core.get_policy(mdp, value_function)


def policy_iteration(mdp, iterations=25):
    value_function = core.get_initial_value_function(mdp)
    policy = core.get_policy(mdp, value_function)

    while True:
        for _ in range(iterations):
            for state in mdp.states:
                state_key = core.key(state)

                current_reward = mdp.get_reward(state)
                future_reward = core.get_action_value(mdp, value_function, state, policy[state_key])
                value_function[state_key] = current_reward + mdp.gamma * future_reward

        new_policy = core.get_policy(mdp, value_function)

        if policy == new_policy:
            return policy

        policy = new_policy
