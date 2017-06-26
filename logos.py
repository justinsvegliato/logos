import solvers


class VI(object):
    def __init__(self, epsilon=0.01):
        self.epsilon = epsilon

    def solve(self, mdp):
        return solvers.vi(mdp, self.epsilon)


class PI(object):
    def __init__(self, iterations=25):
        self.iterations = iterations

    def solve(self, mdp):
        return solvers.pi(mdp, self.iterations)


class MC(object):
    def __init__(self, episodes=100, epsilon=0.1):
        self.episodes = episodes
        self.epsilon = epsilon

    def solve(self, mdp):
        return solvers.mc(mdp, self.episodes, self.epsilon)


class TD(object):
    def __init__(self, episodes=100, step_size=0.01, epsilon=0.1):
        self.episodes = episodes
        self.step_size = step_size
        self.epsilon = epsilon

    def solve(self, mdp):
        return solvers.td(mdp, self.episodes, self.step_size, self.epsilon)


class MDP(object):
    def __init__(self, states, actions, get_transition_probability, get_reward, gamma=0.9):
        self.states = states
        self.actions = actions
        self.get_transition_probability = get_transition_probability
        self.get_reward = get_reward
        self.gamma = gamma

    def solve(self, solver=VI):
        return solver.solve(self)
