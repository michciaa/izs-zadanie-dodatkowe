import matplotlib.pyplot as plt
import numpy as np
import sympy as smp
from scipy.stats import cauchy, norm

M = 1000  # ilość symulacji
N = 100  # liczebność próby
a, b = 0, 1  # przedział [a,b]

Ns = [10, 100, 1000]
ps = [0.1, 0.5, 0.8]


def clt_simulation_uniform_dist(a, b):
    x = smp.symbols("x")
    y = 1 / (b - a)  # funkcja gęstości prawdopodobieństwa
    μ = smp.integrate(x * y, (x, a, b))  # wartość oczekiwana
    μ2 = smp.integrate(x**2 * y, (x, a, b))
    var = μ2 - μ**2  # wariancja
    σ = smp.sqrt(var)  # odchylenie standardowe
    return float(μ), float(σ)


def clt_run_simulation():
    μ, σ = clt_simulation_uniform_dist(a, b)
    σ_clt = σ / np.sqrt(N)

    data = np.random.uniform(a, b, (M, N))  # zmienna losowa
    sample_μ = np.mean(data, axis=1)
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.hist(
        data.flatten(),
        bins=30,
        density=True,
        alpha=0.6,
        color="blue",
        label="Simulation (histogram of random variables)",
    )

    plt.axhline(
        y=1 / (b - a),
        color="r",
        linestyle="--",
        label="Density of Probability Function",
    )

    plt.title(f"Continuous uniform distribution per sample")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.hist(
        sample_μ,
        bins=30,
        density=True,
        alpha=0.6,
        color="red",
        label="Simulation (histogram of sample means)",
    )

    X = np.linspace(min(sample_μ), max(sample_μ), 200)
    plt.plot(
        X,
        norm.pdf(X, μ, σ_clt),
        "r--",
        linewidth=2,
        label="Gauss Probability Density Function",
    )
    plt.title(f"Distribution of means for N = {N}")
    plt.legend()
    plt.grid(alpha=0.3)

    plt.tight_layout()
    plt.show()


def clt_cauchy_simulation_uniform_dist():
    data = np.random.standard_cauchy((M, N))
    sample_μ = np.mean(data, axis=1)

    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    # Clip to range [-10, 10]
    clipped_data = data[0, :][(data[0, :] > -10) & (data[0, :] < 10)]
    plt.hist(
        clipped_data,
        bins=20,
        density=True,
        alpha=0.6,
        color="skyblue",
        label="Histogram of Standard Cauchy Distribution",
    )

    X = np.linspace(-10, 10, 100)
    plt.plot(
        X, cauchy.pdf(X), "r--", label="Simulation (histogram of random variables)"
    )
    plt.title(f"Cauchy Distribution per sample")
    plt.legend()
    clipped_μ = sample_μ[(sample_μ > -10) & (sample_μ < 10)]
    plt.subplot(1, 2, 2)
    plt.hist(
        clipped_μ,
        bins=30,
        density=True,
        alpha=0.6,
        color="red",
        label="Simulation (histogram of sample means) - clipped in range [-10,10]",
    )

    plt.plot(
        X,
        cauchy.pdf(X),
        "k--",
        linewidth=2,
        label="Cauchy Probability Density Function",
    )
    plt.title(f"Distribution of means for N = {N}")
    plt.legend()
    plt.grid(alpha=0.3)

    plt.tight_layout()
    plt.show()


def analyze_params_p_and_N():
    plt.figure(figsize=(12, 5))
    for i, p in enumerate(ps):
        plt.subplot(1, 3, i + 1)
        for N in Ns:
            data = np.random.binomial(n=N, p=p, size=1000)
            plt.hist(data, bins=20, alpha=0.5, density=True, label=f"N = {N}")

        plt.title(f"Influence of p={p}")
        plt.xlabel("Final Position (SUM)")
        plt.legend()
    plt.suptitle("Effect of N (Steps) and p (Probabilities) on Distribution")
    plt.tight_layout()
    plt.show()


clt_run_simulation()
clt_cauchy_simulation_uniform_dist()
analyze_params_p_and_N()
