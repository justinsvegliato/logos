class POMDP(object):
    def __init__(self, get_states, get_actions, get_transition_probabilities, get_reward, get_observations, get_observation_probabilities):
        self.get_states = get_states
        self.get_actions = get_actions
        self.get_transition_probabilities = get_transition_probabilities
        self.get_reward = get_reward
        self.get_observations = get_observations
        self.get_observation_probabilities = get_observation_probabilities
