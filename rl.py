import random

import numpy as np

import core
import grid_world as domain


def monte_carlo(mdp, episodes=250, epsilon=0.1):
    action_value_function = core.get_initial_action_value_function(mdp)
    policy = core.get_epsilon_soft_policy(mdp, action_value_function, epsilon)
    returns = core.get_initial_returns(mdp)

    for _ in range(episodes):
        episode = core.get_episode(mdp, domain, policy)
        action_value_function = core.get_estimated_action_value_function(mdp, action_value_function, episode, returns)
        policy = core.get_epsilon_soft_policy(mdp, action_value_function, epsilon)

    return core.get_deterministic_policy(mdp, policy)


def td_learning(mdp, episodes=250, step_size=0.01, discount_factor=0.9, epsilon=0.1):
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
            current_estimate = reward + discount_factor * action_value_function[next_state_key][next_action]
            previous_estimate = action_value_function[state_key][action]
            error = current_estimate - previous_estimate
            action_value_function[state_key][action] += step_size * error

            state = next_state
            action = next_action

        policy = core.get_epsilon_soft_policy(mdp, action_value_function, epsilon)

    return core.get_deterministic_policy(mdp, policy)
