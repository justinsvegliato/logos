import numpy as np

import computation
from utils import key


def get_partial_policy(ssp, visited_states, values):
    policy = {}

    for state in visited_states.values():
        partial_value_function = {}

        for action in ssp.get_actions(state):
            current_cost = ssp.get_cost(state, action)
            future_cost = computation.get_expected_action_value(ssp, state, action, values)
            partial_value_function[action] = current_cost + future_cost

        policy[key(state)] = np.argmin(ssp.get_actions(state), lambda action: partial_value_function[action])

    return policy


def solve(ssp, trials):
    value_function = computation.get_initial_value_function(ssp)
    visited_states = {}

    for _ in range(trials):
        state = ssp.start_state

        while not np.array_equal(state, ssp.goal_state):
            visited_states[key(state)] = state

            action = computation.get_optimal_action(ssp, state, value_function, maximize=False)
            current_cost = ssp.get_cost(state, action)
            future_cost = min(computation.get_expected_action_values(ssp, state, value_function))
            value_function[key] = current_cost + future_cost

            state = computation.get_sample_state(ssp, state, action)

        visited_states[key(state)] = state

    return get_partial_policy(ssp, visited_states, value_function)
