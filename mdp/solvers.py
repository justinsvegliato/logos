import random
import numpy as np
import utils


def get_initial_values(mdp):
    values = {}
    for state in mdp.states:
        key = mdp.get_key(state)
        values[key] = 0
    return values


def get_initial_policy(mdp):
    policy = {}
    for state in mdp.states:
        key = mdp.get_key(state)
        actions = mdp.get_actions(state)
        policy[key] = random.choice(actions)
    return policy


def get_expected_value(mdp, action, state, values):
    expected_value = 0
    for result_state, probability in mdp.get_transition_probabilities(state, action):
        key = mdp.get_key(result_state)
        expected_value += probability * values[key]
    return expected_value


def get_expected_values(mdp, state, values):
    expected_values = []
    for action in mdp.get_actions(state):
        expected_value = get_expected_value(mdp, action, state, values)
        expected_values.append(expected_value)
    return expected_values


def get_best_action(mdp, state, values, maximize=True):
    get_value = lambda action: get_expected_value(mdp, action, state, values)
    actions = mdp.get_actions(state)
    return utils.argmax(actions, get_value) if maximize else utils.argmin(actions, get_value)


def get_optimal_policy(mdp, values):
    policy = {}
    for state in mdp.states:
        key = mdp.get_key(state)
        policy[key] = get_best_action(mdp, state, values)
    return policy


def evaluate_policy(mdp, policy, values, iterations):
    for _i in range(iterations):
        for state in mdp.states:
            for action in mdp.get_actions(state):
                key = mdp.get_key(state)
                values[key] = mdp.get_reward(state, action) + mdp.gamma * get_expected_value(mdp, policy[key], state,
                                                                                             values)
    return values


def get_improved_policy(mdp, policy, values):
    has_policy_changed = False
    for state in mdp.states:
        action = get_best_action(mdp, state, values)

        key = mdp.get_key(state)
        if action != policy[key]:
            policy[key] = action
            has_policy_changed = True

    return policy if has_policy_changed else None


def get_optimal_values(mdp, epsilon):
    values = get_initial_values(mdp)

    while True:
        new_values = values.copy()
        delta = 0

        for state in mdp.states:
            key = mdp.get_key(state)
            for action in mdp.get_actions(state):
                new_values[key] = mdp.get_reward(state, action) + mdp.gamma * max(
                    get_expected_values(mdp, state, values))
                delta = max(delta, abs(new_values[key] - values[key]))

            values = new_values

        if delta < epsilon * (1 - mdp.gamma) / mdp.gamma:
            return values


def get_partial_policy(ssp, visited_states, values):
    partial_policy = {}
    for state in visited_states.values():
        partial_values = {}
        for action in ssp.get_actions(state):
            partial_values[action] = ssp.get_cost(state, action) + get_expected_value(ssp, action, state, values)

        key = ssp.get_key(state)
        get_value = lambda action: partial_values[action]
        partial_policy[key] = utils.argmin(ssp.get_actions(state), get_value)

    return partial_policy


def value_iteration(mdp, epsilon):
    values = get_optimal_values(mdp, epsilon)
    return get_optimal_policy(mdp, values)


def policy_iteration(mdp, iterations):
    values = get_initial_values(mdp)
    policy = get_initial_policy(mdp)

    while True:
        values = evaluate_policy(mdp, policy, values, iterations)
        new_policy = get_improved_policy(mdp, policy, values)

        if not new_policy:
            return policy

        policy = new_policy


def rtdp(ssp, trials):
    values = get_initial_values(ssp)
    visited_states = {}

    for _i in range(trials):
        current_state = ssp.start_state

        while not np.array_equal(current_state, ssp.goal_state):
            key = ssp.get_key(current_state)
            visited_states[key] = current_state

            best_action = get_best_action(ssp, current_state, values, maximize=False)
            values[key] = ssp.get_cost(current_state, best_action) + min(
                get_expected_values(ssp, current_state, values))

            transition_probabilities = ssp.get_transition_probabilities(current_state, best_action)
            current_state = utils.get_random_variable(transition_probabilities)

        key = ssp.get_key(current_state)
        visited_states[key] = current_state

    return get_partial_policy(ssp, visited_states, values)
