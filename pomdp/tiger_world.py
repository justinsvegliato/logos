STATES = ['tiger-left', 'tiger-right']
ACTIONS = ['open-left', 'open-right', 'listen']

TIGER_LEFT_TRANSITION_PROBABILITIES = {
    'tiger-left': 1.0,
    'tiger-right': 0.0
}
TIGER_RIGHT_TRANSITION_PROBABILITIES = {
    'tiger-left': 0.0,
    'tiger-right': 1.0
}
TRANSITION_PROBABILITIES = {
    'tiger-left': {
        'open-left': TIGER_LEFT_TRANSITION_PROBABILITIES,
        'open-right': TIGER_LEFT_TRANSITION_PROBABILITIES,
        'listen': TIGER_LEFT_TRANSITION_PROBABILITIES
    },
    'tiger-right': {
        'open-left': TIGER_RIGHT_TRANSITION_PROBABILITIES,
        'open-right': TIGER_RIGHT_TRANSITION_PROBABILITIES,
        'listen': TIGER_RIGHT_TRANSITION_PROBABILITIES
    }
}

REWARDS = {
    'tiger-left': {
        'listen': -1,
        'open-left': -100,
        'open-right': 10

    },
    'tiger-right': {
        'listen': -1,
        'open-left': 10,
        'open-right': -100
    }
}

OBSERVATIONS = ['hear-left', 'hear-right']

EMPTY_OBSERVATION_PROBABILITIES = {
    'hear-left': 0.0,
    'hear-right': 0.0
}
OBSERVATION_PROBABILITIES = {
    'tiger-left': {
        'listen': {
            'hear-left': 0.85,
            'hear-right': 0.15
        },
        'open-left': EMPTY_OBSERVATION_PROBABILITIES,
        'open-right': EMPTY_OBSERVATION_PROBABILITIES

    },
    'tiger-right': {
        'listen': {
            'hear-left': 0.15,
            'hear-right': 0.85
        },
        'open-left': EMPTY_OBSERVATION_PROBABILITIES,
        'open-right': EMPTY_OBSERVATION_PROBABILITIES
    }
}

def get_states():
    return STATES

def get_actions():
    return ACTIONS

def get_transition_probabilities(state, action):
    return TRANSITION_PROBABILITIES[state][action]

def get_reward(state, action):
    return REWARDS[state][action]

def get_observations():
    return OBSERVATIONS

def get_observation_probabilities(state, action):
    return OBSERVATION_PROBABILITIES[state][action]
