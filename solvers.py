import random

import numpy as np

import core
import grid_world as domain


def vi(mdp, epsilon):
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


def pi(mdp, iterations):
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


def mc(mdp, episodes, epsilon):
    action_value_function = core.get_initial_action_value_function(mdp)
    policy = core.get_epsilon_soft_policy(mdp, action_value_function, epsilon)
    returns = core.get_initial_returns(mdp)

    for _ in range(episodes):
        episode = core.get_episode(mdp, domain, policy)
        action_value_function = core.get_estimated_action_value_function(mdp, action_value_function, episode, returns)
        policy = core.get_epsilon_soft_policy(mdp, action_value_function, epsilon)

    return core.get_deterministic_policy(mdp, policy)


def td(mdp, episodes, step_size, epsilon):
    action_value_function = core.get_initial_action_value_function(mdp)
    policy = core.get_epsilon_soft_policy(mdp, action_value_function, epsilon)

    for _ in range(episodes):
        state = random.choice(mdp.states)
        action = core.get_sample_action(policy, state)

        while not np.array_equal(state, domain.get_goal_state()):
            next_state = domain.get_next_state(state, action)
            next_action = core.get_sample_action(policy, next_state)

            state_key = core.key(state)
            next_state_key = core.key(next_state)

            reward = mdp.get_reward(state)
            current_estimate = reward + mdp.gamma * action_value_function[next_state_key][next_action]
            previous_estimate = action_value_function[state_key][action]
            error = current_estimate - previous_estimate
            action_value_function[state_key][action] += step_size * error

            state = next_state
            action = next_action

        policy = core.get_epsilon_soft_policy(mdp, action_value_function, epsilon)

    return core.get_deterministic_policy(mdp, policy)
