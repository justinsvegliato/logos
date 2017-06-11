import random

import numpy as np

from utils import key


def get_initial_policy(mdp):
    return {key(state): random.choice(get_actions(mdp, state)) for state in mdp.states}


def get_actions(mdp, state):
    actions = []

    for action in mdp.actions:
        for next_state in mdp.states:
            if mdp.get_transition_probability(state, action, next_state) > 0:
                actions.append(action)
                break

    return actions


def get_initial_value_function(mdp):
    return {key(state): 0 for state in mdp.states}


def get_expected_action_value(mdp, state, action, value_function):
    values = [value_function[key(next_state)] for next_state in mdp.states]
    weights = [mdp.get_transition_probability(state, action, next_state) for next_state in mdp.states]
    return np.average(values, weights=weights)


def get_expected_action_values(mdp, state, value_function):
    return [get_expected_action_value(mdp, state, action, value_function) for action in get_actions(mdp, state)]


def get_optimal_action(mdp, state, value_function, maximize=True):
    actions = get_actions(mdp, state)
    value_function = [get_expected_action_value(mdp, state, action, value_function) for action in actions]
    index = np.argmax(value_function) if maximize else np.argmin(value_function)
    return actions[index]

def get_sample_state(ssp, state, action):
    probabilities = [ssp.get_transition_probability(state, action, next_state) for next_state in ssp.states]
    print(ssp.states)
    return np.random.choice(list(ssp.states), p=probabilities)
    