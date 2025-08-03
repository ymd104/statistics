import numpy as np


def monte_carlo_integrate(f, a, b, n_samples=10000):
    np.random.seed(42)
    samples = np.random.uniform(a, b, n_samples)
    estimate = (b - a) * np.mean(f(samples))

    return estimate


print(monte_carlo_integrate(lambda x: x**x, 0, 1))


def sophomores_dream(n_samples=30):
    total_sum = 0
    for i in range(n_samples):
        term = (-1) ** i / (i + 1) ** (i + 1)
        total_sum += term
    return total_sum


print(sophomores_dream())
