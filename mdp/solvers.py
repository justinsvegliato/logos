import pi
import vi


class VI(object):
    def __init__(self, epsilon=0.01):
        self.epsilon = epsilon

    def solve(self, mdp):
        return vi.solve(mdp, self.epsilon)


class PI(object):
    def __init__(self, iterations=20):
        self.iterations = iterations

    def solve(self, mdp):
        return pi.solve(mdp, self.iterations)


INDEX = {
    'vi': VI,
    'pi': PI
}
