# Author: Pablo Gullith
import os
import numpy as np
import matplotlib.pyplot as plt

IMAGE_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "images"
)
os.makedirs(IMAGE_DIR, exist_ok=True)


# 1. Math functions with diffusion coefficient D
def p(x, t, D=1.0, n_terms=10):
    result = 0.5
    for n in range(1, n_terms + 1):
        result += np.cos(n * np.pi * x) * np.exp(-D * (n**2) * np.pi**2 * t / 2)
    return result


def sigma(t, D=1.0, n_terms=10):
    sum_terms = 0
    for n in range(1, n_terms + 1):
        sum_terms += ((-1) ** n / n**2) * np.exp(-D * (n**2) * np.pi**2 * t / 2)
    variance = np.maximum(1 / 3 + (4 / np.pi**2) * sum_terms, 0)
    return np.sqrt(variance)

# 3. Generate data
x = np.linspace(-1, 1, 200)
t_array = np.linspace(0, 1, 200)
times = [0.05, 0.1, 0.25, 0.5, 1.0]

# 4. Create side-by-side subplots (1 row, 2 columns)
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# First Subplot: p(x,t)
for t in times:
    ax1.plot(x, p(x, t, D=1.0), label=f"1t={t}")

ax1.set(xlim=(-1, 1), ylim=(0, 2), xlabel="$x$", ylabel="$p(x,t)$")
ax1.set_title(f"Probability Density ($D = 1.0$)")
ax1.legend(loc="upper right")

# Second Subplot: sigma(t)
ax2.plot(t_array, sigma(t_array, D=1.0), color="black", label=r"$\sigma(t)$")
ax2.axhline(y=1 / np.sqrt(3), color="red", linestyle="--", label="Asymptote")

ax2.set(xlim=(0, 1), ylim=(0, 1), xlabel="$t$", ylabel=r"$\sigma(t)$")
ax2.set_title(f"Dispersion ($D = 1.0$)")
ax2.legend(loc="lower right")

# 5. Clean up spacing, save and display
plt.tight_layout()
plt.savefig(os.path.join(IMAGE_DIR, f"Figura01.pdf"))
plt.show()
