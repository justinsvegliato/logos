import computation
import numpy as np
import grid_world as domain
from utils import key
import random

# TODO Make sure the goal state is absorbing
# TODO Fix MC logic
# TODO Clean up code

def solve(mdp, episodes, step_size, discount_factor, epsilon):
    action_values = computation.get_initial_action_value_function(mdp)
    policy = computation.get_initial_policy(mdp)

    for _ in range(episodes):
        current_state = domain.get_start_state()
        current_state_key = key(current_state)
        current_action = policy[current_state_key] if random.random() > epsilon else random.choice(mdp.get_actions(current_state))

        visited_states = []

        while not np.array_equal(current_state, domain.get_goal_state()):
            visited_states.append((current_state, False))

            reward = mdp.get_reward(current_state, current_action)

            next_state = domain.get_next_state(current_state, current_action)
            next_state_key = domain.key(next_state)
            next_action = policy[next_state_key] if random.random() > epsilon else random.choice(mdp.get_actions(next_state))

            action_values[current_state_key][current_action] += step_size * (reward + discount_factor * action_values[next_state_key][next_action] - action_values[current_state_key][current_action])

            current_state = next_state
            current_state_key = next_state_key
            current_action = next_action

        policy = improve_policy(mdp, visited_states, action_values, policy)

    return policy