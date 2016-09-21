from solvers import value_iteration, policy_iteration, rtdp

class VI(object):
    def __init__(self, epsilon=0.01):
        self.epsilon = epsilon

    def solve(self, mdp):
        return value_iteration(mdp, self.epsilon)

class PI(object):
    def __init__(self, iterations=25):
        self.iterations = iterations

    def solve(self, mdp):
        return policy_iteration(mdp, self.iterations)

class RTDP(object):
    def __init__(self, trials=25):
        self.trials = trials

    def solve(self, ssp):
        return rtdp(ssp, self.trials)

SOLVERS = {
    'vi': VI,
    'pi': PI,
    'rtdp': RTDP
}

class MDP(object):
    def __init__(self, states, get_actions, get_transition_probabilities, get_reward, get_key, gamma=0.9):
        self.states = states
        self.get_actions = get_actions
        self.get_transition_probabilities = get_transition_probabilities
        self.get_reward = get_reward
        self.get_key = get_key
        self.gamma = gamma

    def solve(self, solver='vi'):
        if isinstance(solver, basestring):
            solver = SOLVERS[solver]()
        return solver.solve(self)

class SSP(object):
    def __init__(self, states, get_actions, get_transition_probabilities, get_cost, get_key, start_state, goal_state):
        self.states = states
        self.get_actions = get_actions
        self.get_transition_probabilities = get_transition_probabilities
        self.get_cost = get_cost 
        self.get_key = get_key
        self.start_state = start_state
        self.goal_state = goal_state

    def solve(self, solver='rtdp'):
        if isinstance(solver, basestring):
            solver = SOLVERS[solver]()
        return solver.solve(self)