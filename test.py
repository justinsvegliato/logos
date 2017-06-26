#!/usr/bin/env python
import json
import time

import numpy as np

import core
import grid_world as domain
from logos import MC, MDP, PI, TD, VI


def simulate(policy):
    state = domain.get_start_state()
    steps = 0

    while not np.array_equal(state, domain.get_goal_state()):
        state_key = core.key(state)
        action = policy[state_key]
        state = domain.get_next_state(state, action)

        steps += 1

        if steps > 1000:
            return False

    return steps


def test(mdp, solver, tests=20):
    start = time.clock()
    policy = mdp.solve(solver=solver())
    end = time.clock()

    steps = [simulate(policy) for test in range(tests)]
    average_steps = np.average(steps)

    return json.dumps({
        'time': end - start,
        'steps': average_steps
    })


def main():
    mdp = MDP(
        domain.get_states(),
        domain.get_actions(),
        domain.get_transition_probability,
        domain.get_reward
    )

    print('VI: %s' % test(mdp, VI))
    print('PI: %s' % test(mdp, PI))
    print('MC: %s' % test(mdp, MC))
    print('TD: %s' % test(mdp, TD))


if __name__ == '__main__':
    main()
