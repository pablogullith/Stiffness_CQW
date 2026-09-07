"""
Reproduces Fig. 2(i) of "Stiffness in Confined Quantum Walks"
(Dantas, Vieira, Amorim).
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm


# Quantum walk simulation


def run_confined_walk(N, t_max):
    """
    Hadamard quantum walk confined to N = 2a+1 sites (j = -a, ..., a)
    with reflecting NOT-gate boundaries [Eqs. (10)-(13) of the paper].

    Initial coin state (alpha, beta) = (pi/2, pi/2):
        |chi> = (1/sqrt2) |L> + i (1/sqrt2) |R>,
    localized at the central site j = 0 [Eq. (16)].

    Returns sigma(t) for t = 0, 1, ..., t_max.
    """
    a = (N - 1) // 2
    positions = np.arange(-a, a + 1, dtype=float)

    psiL = np.zeros(N, dtype=complex)
    psiR = np.zeros(N, dtype=complex)
    psiL[a] = 1 / np.sqrt(2)
    psiR[a] = 1j / np.sqrt(2)

    sigma = np.empty(t_max + 1)

    def sigma_of(pL, pR):
        P = np.abs(pL) ** 2 + np.abs(pR) ** 2
        mx = np.sum(positions * P)
        mx2 = np.sum(positions ** 2 * P)
        return np.sqrt(max(mx2 - mx ** 2, 0.0))

    sigma[0] = sigma_of(psiL, psiR)
    inv_sqrt2 = 1 / np.sqrt(2)

    for t in range(1, t_max + 1):
        # Hadamard coin: C = (1/sqrt2)[[1,1],[1,-1]]
        cL = inv_sqrt2 * (psiL + psiR)
        cR = inv_sqrt2 * (psiL - psiR)

        # Conditional shift with reflecting NOT-gate boundaries
        newL = np.zeros(N, dtype=complex)
        newR = np.zeros(N, dtype=complex)
        newL[:-1] = cL[1:]     # bulk: |L,j> -> |L,j-1>
        newR[1:] = cR[:-1]     # bulk: |R,j> -> |R,j+1>
        newR[0] += cL[0]       # left boundary:  |L,-a> -> |R,-a>
        newL[-1] += cR[-1]     # right boundary: |R, a> -> |L, a>

        psiL, psiR = newL, newR
        sigma[t] = sigma_of(psiL, psiR)

    return sigma



# Reproduce Fig. 2(i)
N_values = [101, 201, 301]
colors = {101: "black", 201: "crimson", 301: "royalblue"}
t_max = 100_000     # up to 10^5 time steps, as in the paper
n_tail = 80_000      # histogram from the last 8x10^4 time steps

fig, ax = plt.subplots(figsize=(6, 4.5))

for N in N_values:
    sigma = run_confined_walk(N, t_max)
    tail = sigma[-n_tail:]

    # empirical PDF (histogram)
    ax.hist(tail, bins=60, density=True, alpha=0.35, color=colors[N])

    # Gaussian fit
    mu, std = norm.fit(tail)
    x = np.linspace(tail.min(), tail.max(), 400)
    ax.plot(x, norm.pdf(x, mu, std), color=colors[N], lw=2, label=f"N={N}")

    sigma_uniform = np.sqrt((N ** 2 - 1) / 12)
    print(f"N={N:4d}:  <sigma> = {mu:6.2f}   std = {std:5.2f}   "
          f"sigma_uniform (ref.) = {sigma_uniform:6.2f}")

ax.set_xlabel(r"$\sigma(t)$")
ax.set_ylabel("PDF")
ax.set_title(r"Histograms of $\sigma(t)$ with Gaussian fits")
ax.legend(frameon=False)
fig.tight_layout()
fig.savefig("fig2_panel_i.png", dpi=200)
plt.show()