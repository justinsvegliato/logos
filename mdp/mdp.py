import solvers


class MDP(object):
    def __init__(self, states, actions, get_transition_probability, get_reward, gamma=0.9):
        self.states = states
        self.actions = actions
        self.get_transition_probability = get_transition_probability
        self.get_reward = get_reward
        self.gamma = gamma

    def solve(self, solver='vi'):
        if isinstance(solver, str):
            solver = solvers.INDEX[solver]()
        return solver.solve(self)
