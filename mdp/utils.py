import random


def argmax(args, function):
    best = args[0]
    best_score = float('-inf')
    for arg in args:
        score = function(arg)
        if score > best_score:
            best, best_score = arg, score
    return best


def argmin(args, function):
    best = args[0]
    best_score = float('inf')
    for arg in args:
        score = function(arg)
        if score < best_score:
            best, best_score = arg, score
    return best


def get_random_variable(probability_distribution):
    selection = random.random()
    total = 0
    for variable, probability in probability_distribution:
        if total + probability >= selection:
            return variable
        total += probability
