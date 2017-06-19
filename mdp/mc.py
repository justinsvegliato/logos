import random

import numpy as np

import computation
import grid_world as domain
from utils import key


def get_episode(mdp, policy, epsilon):
    episode = []

    state = domain.get_start_state()

    while not np.array_equal(state, domain.get_goal_state()):
        state_key = key(state)
        actions = computation.get_actions(mdp, state)
        action = policy[state_key] if random.random() > epsilon else random.choice(actions)

        episode.append((state, action))

        state = domain.get_next_state(state, action)

    episode.append((state, policy[state_key]))

    return episode


def get_action_value_function(mdp, episode, returns, action_value_function):
    new_action_value_function = action_value_function.copy()

    current_return = 0

    for state, action in reversed(episode):
        state_key = key(state)

        current_return += mdp.get_reward(state)
        returns[state_key][action].append(current_return)

        new_action_value_function[state_key][action] = np.average(returns[state_key][action])

    return new_action_value_function


def get_policy(episode, action_value_function, policy):
    new_policy = policy.copy()

    for state, _ in episode:
        state_key = key(state)
        actions = action_value_function[state_key]
        new_policy[state_key] = max(actions, key=actions.get)

    return new_policy


def solve(mdp, episodes, epsilon):
    action_value_function = computation.get_initial_action_value_function(mdp)
    returns = computation.get_initial_returns(mdp)
    policy = computation.get_initial_policy(mdp)

    for _ in range(episodes):
        episode = get_episode(mdp, policy, epsilon)
        action_value_function = get_action_value_function(mdp, episode, returns, action_value_function)
        policy = get_policy(episode, action_value_function, policy)

    return policy
