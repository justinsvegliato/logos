import random
import numpy as np
from utils import key

GRID_SIZE = 3
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
    location = get_location(state)

    x, y = location
    next_x, next_y = get_next_location(location, action)

    next_state = np.asmatrix(np.copy(state))
    next_state[x, y] = 0
    next_state[next_x, next_y] = 1

    return next_state


def get_next_state(state, action):
    return state if random.random() <= SLIP_PROBABILITY else get_successor_state(state, action)


def get_location(state):
    locations = np.where(state == ROBOT_SYMBOL)
    x = locations[0][0]
    y = locations[1][0]
    return x, y


def get_next_location(location, action):
    x, y = location
    delta_x, delta_y = ACTIONS[action]
    new_x, new_y = x + delta_x, y + delta_y
    return new_x, new_y


def is_valid_location(location):
    x, y = location
    return GRID_SIZE > x >= 0 and GRID_SIZE > y >= 0


def get_successor_states(state):
    successor_states = []

    location = get_location(state)
    for action in ACTIONS:
        next_location = get_next_location(location, action)
        if is_valid_location(next_location):
            successor_state = get_successor_state(state, action)
            successor_states.append(successor_state)

    return successor_states


def get_states():
    states = {}

    frontier = [get_start_state()]
    while len(frontier) > 0:
        current_state = frontier.pop()

        states[key(current_state)] = current_state

        for state in get_successor_states(current_state):
            if key(state) not in states:
                states[key(state)] = state
                frontier.append(state)

    return states.values()


def get_actions():
    return ACTIONS


def get_transition_probability(state, action, next_state):
    current_location = get_location(state)
    target_location = get_location(next_state)
    next_location = get_next_location(current_location, action)

    if target_location != next_location:
        return 0

    if current_location == target_location:
        return SLIP_PROBABILITY

    return 1 - SLIP_PROBABILITY


def get_reward(state):
    return GOAL_REWARD if np.array_equal(state, get_goal_state()) else NON_GOAL_REWARD


def get_cost(state, action):
    if np.array_equal(state, get_goal_state()) and action == ACTIONS['Stay']:
        return GOAL_COST
    return NON_GOAL_COST
