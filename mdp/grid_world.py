import random
import numpy as np

GRID_SIZE = 6
SLIP_PROBABILITY = 0.1

GOAL_REWARD = 1
NON_GOAL_REWARD = -0.04

GOAL_COST = 0
NON_GOAL_COST = 0.04

EMPTY_SYMBOL = 0
ROBOT_SYMBOL = 1
SYMBOLS = ['O', 'R']

ACTIONS = {
    'North': (0, 1),
    'East': (1, 0),
    'South': (0, -1),
    'West': (-1, 0),
    'Stay': (0, 0)
}


def get_base_state():
    return np.asmatrix(np.zeros((GRID_SIZE, GRID_SIZE))).astype(int)


def get_start_state():
    state = get_base_state()
    state[0, 0] = ROBOT_SYMBOL
    return state


def get_goal_state():
    state = get_base_state()
    state[GRID_SIZE - 1, GRID_SIZE - 1] = ROBOT_SYMBOL
    return state


def get_successor_state(state, action):
    x, y = get_robot_location(state)

    delta_x, delta_y = ACTIONS[action]
    new_x, new_y = x + delta_x, y + delta_y

    new_state = np.asmatrix(np.copy(state))
    new_state[x, y] = 0
    new_state[new_x, new_y] = 1

    return new_state


def get_next_state(state, action):
    return state if random.random() <= SLIP_PROBABILITY else get_successor_state(state, action)


def get_robot_location(state):
    locations = np.where(state == ROBOT_SYMBOL)
    x = locations[0][0]
    y = locations[1][0]
    return x, y


def is_valid_location(x, y):
    return GRID_SIZE > x >= 0 and GRID_SIZE > y >= 0


def get_successor_states(state):
    successor_states = []
    for action in get_actions(state):
        successor_states.append(get_successor_state(state, action))
    return successor_states


def get_key(state):
    representation = ''
    for row in np.array(state):
        for element in row:
            representation += str(element)
    return representation


def get_states():
    states = {}

    frontier = [get_start_state()]
    while len(frontier) > 0:
        current_state = frontier.pop()

        current_state_key = get_key(current_state)
        states[current_state_key] = current_state

        for state in get_successor_states(current_state):
            successor_state_key = get_key(state)
            if successor_state_key in states:
                continue

            states[successor_state_key] = state
            frontier.append(state)

    return states.values()


def get_actions(state):
    x, y = get_robot_location(state)

    actions = []
    for action in ACTIONS:
        delta_x, delta_y = ACTIONS[action]
        new_x, new_y = x + delta_x, y + delta_y

        if is_valid_location(new_x, new_y):
            actions.append(action)

    return actions


def get_transition_probabilities(state, action):
    return [
        (state, SLIP_PROBABILITY),
        (get_successor_state(state, action), 1 - SLIP_PROBABILITY)
    ]


def get_reward(state, action):
    return GOAL_REWARD if np.array_equal(state, get_goal_state()) else NON_GOAL_REWARD


def get_cost(state, action):
    if np.array_equal(state, get_goal_state()) and action == ACTIONS['Stay']:
        return GOAL_COST
    return NON_GOAL_COST
