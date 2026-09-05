#Autor: Pablo Gullith
import numpy as np
import matplotlib.pyplot as plt

# 1. Math functions
def p(x, t, n_terms=10):
    result = 0.5
    for n in range(1, n_terms + 1):
        result += np.cos(n * np.pi * x) * np.exp(-n**2 * np.pi**2 * t / 2)
    return result

def sigma(t, n_terms=10):
    sum_terms = 0
    for n in range(1, n_terms + 1):
        sum_terms += ((-1)**n / n**2) * np.exp(-n**2 * np.pi**2 * t / 2)
    return np.sqrt(1/3 + (4 / np.pi**2) * sum_terms)

# 2. Generate data
x = np.linspace(-1, 1, 200)
t_array = np.linspace(0, 1, 200)
times = [0.05, 0.1, 0.25, 0.5, 1.0]

# 3. Create side-by-side subplots (1 row, 2 columns)
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# First Subplot: p(x,t)
for t in times:
    ax1.plot(x, p(x, t), label=f"t={t}")

ax1.set(xlim=(-1, 1), ylim=(0, 2), xlabel='$x$', ylabel='$p(x,t)$')
ax1.set_title('Probability Density')
ax1.legend(loc='upper right')

# Second Subplot: sigma(t)
ax2.plot(t_array, sigma(t_array), color='black', label=r'$\sigma(t)$')
ax2.axhline(y=1/np.sqrt(3), color='red', linestyle='--', label='Asymptote')

ax2.set(xlim=(0, 1), ylim=(0, 1), xlabel='$t$', ylabel=r'$\sigma(t)$')
ax2.set_title('Dispersion')
ax2.legend(loc='lower right')

# 4. Clean up spacing and display
plt.tight_layout()
plt.savefig('Figura01.png', dpi=300)
plt.savefig('Figura01.pdf')
plt.show()
