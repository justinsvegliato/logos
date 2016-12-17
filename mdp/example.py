#!/usr/bin/env python
import time
import grid_world as domain
from mdp import MDP, VI, PI, SSP, RTDP
import numpy as np


def execute_policy(policy):
    steps = 0
    current_state = domain.get_start_state()

    while not np.array_equal(current_state, domain.get_goal_state()):
        key = domain.get_key(current_state)
        current_state = domain.get_next_state(current_state, policy[key])

        steps += 1

        if steps > 10000:
            return False

    return steps


def execute_vi_example(mdp):
    vi = VI(epsilon=0.1)

    t0 = time.clock()
    policy = mdp.solve(solver=vi)
    t1 = time.clock()
    print 'The MDP was solved using VI in %d seconds.' % (t1 - t0)

    steps = execute_policy(policy)
    print 'The agent reached the goal in %s steps.' % steps


def execute_pi_example(mdp):
    pi = PI(iterations=20)

    t0 = time.clock()
    policy = mdp.solve(solver=pi)
    t1 = time.clock()
    print 'The MDP was solved using PI in %d seconds.' % (t1 - t0)

    steps = execute_policy(policy)
    print 'The agent reached the goal in %s steps.' % steps


def execute_rtdp_example(ssp):
    rtdp = RTDP(trials=100)

    t0 = time.clock()
    policy = ssp.solve(solver=rtdp)
    t1 = time.clock()
    print 'The MDP was solved using RTDP in %d seconds.' % (t1 - t0)

    steps = execute_policy(policy)
    print 'The agent reached the goal in %s steps.' % steps


def main():
    mdp = MDP(
        domain.get_states(),
        domain.get_actions,
        domain.get_transition_probabilities,
        domain.get_reward,
        domain.get_key
    )
    execute_vi_example(mdp)
    execute_pi_example(mdp)

    ssp = SSP(
        domain.get_states(),
        domain.get_actions,
        domain.get_transition_probabilities,
        domain.get_cost,
        domain.get_key,
        domain.get_start_state(),
        domain.get_goal_state()
    )
    execute_rtdp_example(ssp)


if __name__ == '__main__':
    main()
