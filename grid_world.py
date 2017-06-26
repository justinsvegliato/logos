import random

import numpy as np

from core import key

GRID_SIZE = 3

START_ROW = 0
START_COLUMN = 0

GOAL_ROW = GRID_SIZE - 1
GOAL_COLUMN = GRID_SIZE - 1

SLIP_PROBABILITY = 0.1

GOAL_REWARD = 1
NON_GOAL_REWARD = -0.04

ROBOT_SYMBOL = 1

ACTIONS = {
    'North': (-1, 0),
    'East': (0, 1),
    'South': (1, 0),
    'West': (0, -1),
    'Stay': (0, 0)
}


def get_base_state():
    return np.asmatrix(np.zeros((GRID_SIZE, GRID_SIZE))).astype(int)


def get_start_state():
    state = get_base_state()
    state[START_ROW, START_COLUMN] = ROBOT_SYMBOL
    return state


def get_goal_state():
    state = get_base_state()
    state[GOAL_ROW, GOAL_COLUMN] = ROBOT_SYMBOL
    return state


def get_successor_state(state, action):
    location = get_location(state)
    next_location = get_next_location(location, action)

    if not is_valid_location(next_location):
        return state

    current_x, current_y = location
    next_x, next_y = next_location

    next_state = np.asmatrix(np.copy(state))
    next_state[current_x, current_y] = 0
    next_state[next_x, next_y] = 1

    return next_state


def get_next_state(state, action):
    return state if random.random() <= SLIP_PROBABILITY else get_successor_state(state, action)


def get_location(state):
    locations = np.where(state == ROBOT_SYMBOL)
    current_x = locations[0][0]
    current_y = locations[1][0]
    return current_x, current_y


def get_next_location(location, action):
    x, y = location
    delta_x, delta_y = ACTIONS[action]
    new_x, new_y = x + delta_x, y + delta_y
    return new_x, new_y


def is_valid_location(location):
    current_x, current_y = location
    return GRID_SIZE > current_x >= 0 and GRID_SIZE > current_y >= 0


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

    return list(states.values())


def get_actions():
    return ACTIONS


def get_transition_probability(state, action, next_state):
    current_location = get_location(state)
    target_location = get_location(next_state)
    next_location = get_next_location(current_location, action)
    goal_location = get_goal_state()

    if current_location == goal_location:
        return 1

    if current_location == target_location:
        return SLIP_PROBABILITY

    if target_location != next_location:
        return 0

    states = get_successor_states(state)
    normalizer = len(states) - 1
    return (1 - SLIP_PROBABILITY) / normalizer


def get_reward(state):
    return GOAL_REWARD if np.array_equal(state, get_goal_state()) else NON_GOAL_REWARD
