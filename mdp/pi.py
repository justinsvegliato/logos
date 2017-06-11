import random

import computation
from utils import key


def get_value_function(mdp, policy, value_function, iterations):
    new_value_function = value_function.copy()

    for _ in range(iterations):
        for state in mdp.states:
            state_key = key(state)

            current_reward = mdp.get_reward(state)
            future_reward = computation.get_expected_action_value(mdp, state, policy[state_key], value_function)
            new_value_function[state_key] = current_reward + mdp.gamma * future_reward

    return new_value_function


def get_policy(mdp, policy, value_function):
    return {key(state): computation.get_optimal_action(mdp, state, value_function) for state in mdp.states}


def solve(mdp, iterations):
    new_value_function = computation.get_initial_value_function(mdp)
    policy = computation.get_initial_policy(mdp)

    while True:
        new_value_function = get_value_function(mdp, policy, new_value_function, iterations)
        new_policy = get_policy(mdp, policy, new_value_function)

        if policy == new_policy:
            return policy

        policy = new_policy
