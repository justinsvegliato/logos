import numpy as np

import computation
from utils import key


def get_optimal_policy(mdp, value_function):
    return {key(state): computation.get_optimal_action(mdp, state, value_function) for state in mdp.states}


def get_optimal_value_function(mdp, epsilon):
    value_function = computation.get_initial_value_function(mdp)

    while True:
        new_value_function = value_function.copy()

        delta = 0

        for state in mdp.states:
            state_key = key(state)

            current_reward = mdp.get_reward(state)
            future_reward = max(computation.get_expected_action_values(mdp, state, value_function))
            new_value_function[state_key] = current_reward + mdp.gamma * future_reward

            delta = max(delta, abs(new_value_function[state_key] - value_function[state_key]))

        value_function = new_value_function

        if delta < epsilon:
            return value_function


def solve(mdp, epsilon):
    value_function = get_optimal_value_function(mdp, epsilon)
    return get_optimal_policy(mdp, value_function)
