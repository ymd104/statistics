from scipy.stats import norm

x = [1.3, 1.5, 1.6, 1.7, 4.7, 5.0, 5.2, 5.5]
n = len(x)

pi = 0.5
mu_1 = 1
mu_2 = 6
sigma_1 = 1
sigma_2 = 1

for i in range(20):

    gamma_1 = []
    gamma_2 = []
    for xi in x:
        gamma_1.append(
            pi
            * norm.pdf(xi, mu_1, sigma_1)
            / (
                pi * norm.pdf(xi, mu_1, sigma_1)
                + (1 - pi) * norm.pdf(xi, mu_2, sigma_2)
            )
        )
        gamma_2.append(
            (1 - pi)
            * norm.pdf(xi, mu_2, sigma_2)
            / (
                pi * norm.pdf(xi, mu_1, sigma_1)
                + (1 - pi) * norm.pdf(xi, mu_2, sigma_2)
            )
        )

    pi = sum(gamma_1) / n
    mu_1 = sum([x * y for (x, y) in zip(gamma_1, x)]) / sum(gamma_1)
    mu_2 = sum([x * y for (x, y) in zip(gamma_2, x)]) / sum(gamma_2)

    print(i + 1, pi, mu_1, mu_2)

print(gamma_1)
print(gamma_2)
