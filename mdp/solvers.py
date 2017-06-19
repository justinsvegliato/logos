import mc
import pi
import vi
import td


class VI(object):
    def __init__(self, epsilon=0.01):
        self.epsilon = epsilon

    def solve(self, mdp):
        return vi.solve(mdp, self.epsilon)


class PI(object):
    def __init__(self, iterations=25):
        self.iterations = iterations

    def solve(self, mdp):
        return pi.solve(mdp, self.iterations)


class MC(object):
    def __init__(self, episodes=50, epsilon=0.1):
        self.episodes = episodes
        self.epsilon = epsilon

    def solve(self, mdp):
        return mc.solve(mdp, self.episodes, self.epsilon)


class TD(object):
    def __init__(self, episodes=50, epsilon=0.1):
        self.episodes = episodes
        self.epsilon = epsilon

    def solve(self, mdp):
        return td.solve(mdp, self.episodes, self.epsilon)


INDEX = {
    'vi': VI,
    'pi': PI,
    'mc': MC,
    'td': TD
}
