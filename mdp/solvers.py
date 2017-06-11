import pi
import rtdp
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

class RTDP(object):
    def __init__(self, trials=20):
        self.trials = trials

    def solve(self, ssp):
        return rtdp.solve(ssp, self.trials)


INDEX = {
    'vi': VI,
    'pi': PI,
    'rtdp': RTDP
}
