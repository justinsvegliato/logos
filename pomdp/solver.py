def get_initial_values(states):
    return [0] * len(states)

def get_reward(pomdp, belief_state, action):
    reward = 0
    for state in pomdp.get_states():
        reward += belief_state[state] * pomdp.get_reward(state, action)
    return reward

def get_conditioned_observation_probability(pomdp, observation, belief_state, action):
    probability = 0
    for result_state in pomdp.get_states():
        outer_probability = pomdp.get_observation_probabilities(result_state, action)[observation]

        inner_probability = 0
        for state in pomdp.get_states():
            transition_probability = pomdp.get_transition_probabilities(state, action)[result_state]
            state_probability = belief_state[state]
            inner_probability += transition_probability * state_probability

        probability += outer_probability * inner_probability

    return probability


def solve(pomdp, steps):
    values = get_initial_values(pomdp.get_states())
    t = 0

      
