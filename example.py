#!/usr/bin/env python
import json
import time

import numpy as np

import core
import dp
import grid_world as domain
import rl
from mdp import MDP


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
    policy = solver(mdp)
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

    print('VI: %s' % test(mdp, dp.value_iteration))
    print('PI: %s' % test(mdp, dp.policy_iteration))
    print('MC: %s' % test(mdp, rl.monte_carlo))
    print('TD: %s' % test(mdp, rl.td_learning))


if __name__ == '__main__':
    main()
