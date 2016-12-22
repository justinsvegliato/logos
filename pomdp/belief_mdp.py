class BeliefMDP(object):
    def __init__(self, belief_states, get_actions, get_reward, get_transition_probabilities):
        self.belief_states = belief_states
        self.get_actions = get_actions
        self.get_reward = get_reward
        self.get_transition_probabilities = get_transition_probabilities
