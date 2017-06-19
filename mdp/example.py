#!/usr/bin/env python
import json
import time

import numpy as np

import grid_world as domain
from mdp import MDP
from utils import key


def simulate(policy):
    state = domain.get_start_state()
    steps = 0

    while not np.array_equal(state, domain.get_goal_state()):
        state = domain.get_next_state(state, policy[key(state)])

        steps += 1

        if steps > 1000:
            return False

    return steps


def test(mdp, solver, tests=20):
    start = time.clock()
    policy = mdp.solve(solver=solver)
    end = time.clock()

    steps = [simulate(policy) for test in range(tests)]
    average_steps = np.average(steps)

    results = json.dumps({
        'solver': solver,
        'time': end - start,
        'steps': average_steps
    })

    print(results)


def main():
    mdp = MDP(
        domain.get_states(),
        domain.get_actions(),
        domain.get_transition_probability,
        domain.get_reward
    )

    test(mdp, 'vi')
    test(mdp, 'pi')
    test(mdp, 'mc')


if __name__ == '__main__':
    main()
