import random
import grid_world as domain
import numpy as np


def get_initial_policy(mdp):
    return {mdp.get_key(state): random.choice(mdp.get_actions(state)) for state in mdp.states}


def get_episode(mdp, policy, epsilon):
    episode = []

    current_state = domain.get_start_state()
    key = domain.key(current_state)
    current_action = policy[key] if random.random() > epsilon else random.choice(mdp.get_actions(current_state))
    episode.append((current_state, current_action))

    current_state = domain.get_next_state(current_state, current_action)

    while not np.array_equal(current_state, domain.get_goal_state()):
        key = domain.key(current_state)
        current_action = policy[key] if random.random() > epsilon else random.choice(mdp.get_actions(current_state))
        episode.append((current_state, current_action))

        current_state = domain.get_next_state(current_state, current_action)

    return episode


def evaluate_policy(mdp, episode, returns, action_values):
    updated_action_values = action_values.copy()

    current_return = 0

    for pair in reversed(episode):
        state, action = pair[0], pair[1]
        key = mdp.get_key(state)

        current_return += mdp.get_reward(state, action)

        if key not in returns:
            returns[key] = {}

        if action not in returns[key]:
            returns[key][action] = []

        returns[key][action].append(current_return)

        if key not in updated_action_values:
            updated_action_values[key] = {}

        updated_action_values[key][action] = np.average(returns[key][action])

    return updated_action_values


def improve_policy(mdp, episode, action_values, policy):
    improved_policy = policy.copy()

    for pair in episode:
        key = mdp.get_key(pair[0])
        improved_policy[key] = max(action_values[key], key=action_values[key].get)

    return improved_policy


def monte_carlo(mdp, episodes, epsilon):
    action_values = {}
    policy = get_initial_policy(mdp)
    returns = {}

    for _ in xrange(episodes):
        episode = get_episode(mdp, policy, epsilon)
        action_values = evaluate_policy(mdp, episode, returns, action_values)
        policy = improve_policy(mdp, episode, action_values, policy)

    return policy


def td_learning(mdp, episodes, step_size, discount_factor, epsilon):
    action_values = {}
    policy = get_initial_policy(mdp)

    for _ in xrange(episodes):
        current_state = domain.get_start_state()
        current_state_key = domain.key(current_state)
        current_action = policy[current_state_key] if random.random() > epsilon else random.choice(mdp.get_actions(current_state))

        visited_states = []

        while not np.array_equal(current_state, domain.get_goal_state()):
            visited_states.append((current_state, False))

            reward = mdp.get_reward(current_state, current_action)

            next_state = domain.get_next_state(current_state, current_action)
            next_state_key = domain.key(next_state)
            next_action = policy[next_state_key] if random.random() > epsilon else random.choice(mdp.get_actions(next_state))

            if current_state_key not in action_values:
                action_values[current_state_key] = {}

            if current_action not in action_values[current_state_key]:
                action_values[current_state_key][current_action] = 0

            if next_state_key not in action_values:
                action_values[next_state_key] = {}

            if next_action not in action_values[next_state_key]:
                action_values[next_state_key][next_action] = 0

            action_values[current_state_key][current_action] += step_size * (reward + discount_factor * action_values[next_state_key][next_action] - action_values[current_state_key][current_action])

            current_state = next_state
            current_state_key = next_state_key
            current_action = next_action

        policy = improve_policy(mdp, visited_states, action_values, policy)

    return policy
