import random

import numpy as np


def key(state):
    return str(state.tolist())


def get_initial_value_function(mdp):
    return {key(state): 0 for state in mdp.states}


def get_initial_action_value_function(mdp):
    return {key(state): {action: random.random() for action in mdp.actions} for state in mdp.states}


def get_initial_returns(mdp):
    return {key(state): {action: [] for action in mdp.actions} for state in mdp.states}


def get_action_value_function(mdp, value_function):
    return {key(state): {action: get_action_value(mdp, value_function, state, action) for action in mdp.actions} for state in mdp.states}


def get_estimated_action_value_function(mdp, action_value_function, episode, returns):
    new_action_value_function = action_value_function.copy()

    total_reward = 0

    for state, action in reversed(episode):
        state_key = key(state)

        total_reward += mdp.get_reward(state)
        returns[state_key][action].append(total_reward)

        new_action_value_function[state_key][action] = np.average(returns[state_key][action])

    return new_action_value_function


def get_action_values(mdp, value_function, state):
    return [get_action_value(mdp, value_function, state, action) for action in get_actions(mdp, state)]


def get_action_value(mdp, value_function, state, action):
    values = [value_function[key(next_state)] for next_state in mdp.states]
    weights = [mdp.get_transition_probability(state, action, next_state) for next_state in mdp.states]
    return np.average(values, weights=weights)


def get_actions(mdp, state):
    actions = []

    for action in mdp.actions:
        for next_state in mdp.states:
            if mdp.get_transition_probability(state, action, next_state) > 0:
                actions.append(action)
                break

    return actions


def get_greedy_action(action_value_function, state):
    return max(action_value_function[key(state)], key=action_value_function[key(state)].get)


def get_most_probable_action(policy, state):
    return max(policy[key(state)], key=lambda action: action[1])[0]


def get_action_distribution(mdp, action_value_function, state, epsilon):
    actions = get_actions(mdp, state)
    greedy_action = get_greedy_action(action_value_function, state)

    exploitation_probability = 1 - epsilon
    exploration_probability = epsilon / (len(actions) - 1)

    action_distribution = [(greedy_action, exploitation_probability)]
    for action in actions:
        if action != greedy_action:
            action_distribution.append((action, exploration_probability))

    return action_distribution


def get_policy(mdp, value_function):
    action_value_function = get_action_value_function(mdp, value_function)
    return {key(state): get_greedy_action(action_value_function, state) for state in mdp.states}


def get_epsilon_soft_policy(mdp, action_value_function, epsilon):
    return {key(state): get_action_distribution(mdp, action_value_function, state, epsilon) for state in mdp.states}


def get_deterministic_policy(mdp, policy):
    return {key(state): get_most_probable_action(policy, state) for state in mdp.states}


def get_sample_action(policy, state):
    state_key = key(state)
    action_distribution = policy[state_key]
    actions, probabilities = zip(*action_distribution)
    return np.random.choice(actions, p=probabilities)


def get_episode(mdp, domain, policy):
    episode = []

    state = random.choice(mdp.states)
    action = get_sample_action(policy, state)

    while not np.array_equal(state, domain.get_goal_state()):
        episode.append((state, action))

        state = domain.get_next_state(state, action)
        action = get_sample_action(policy, state)

    episode.append((state, action))

    return episode
