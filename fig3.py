r"""
FIG. 3
'Stiffness in Confined Quantum Walks'
Pablo G. M. Dantas, Rafael Vieira, Edgard P. M. Amorim

Time evolution of the spreading stiffness S_\nu(t) for the localized (black),
Gaussian with \sigma_0 = 3 (red), and truncated-uniform (blue) initial states for N = 1001.
The dashed vertical line at t_c = 2 x 10^4 marks the end of the transient interval
discarded in the long-time averages. The inset shows the initial dynamics over t <= t_c,
revealing the coherent oscillatory structure underlying the dense long-time bands
observed in the main panel.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import matplotlib.ticker as ticker
from numba import njit

CACHE_FILE = "fig3_cache.npz"


def setup_matplotlib_style():
    """Configures the style identical to the paper (Computer Modern / STIX serif)."""
    plt.rcParams.update({
        "font.family": "serif",
        "mathtext.fontset": "stix",
        "font.size": 12,
        "axes.linewidth": 1.0,
        "xtick.direction": "in",
        "ytick.direction": "in",
        "xtick.top": True,
        "ytick.right": True,
    })


@njit(fastmath=True)
def _simulate_qw_sigma(psiL_init, psiR_init, j_vals, t_max):
    """
    Numba-optimized unitary evolution of the confined discrete-time quantum walk.
    Lattice: j in [-(a+1), a+1] with Hadamard bulk in [-a, a] and Pauli-X (NOT) boundaries at +/- (a+1).
    Returns the standard deviation sigma(t) for t = 0, ..., t_max.
    """
    N_full = len(psiL_init)
    psiL = psiL_init.copy()
    psiR = psiR_init.copy()
    sigma = np.zeros(t_max + 1, dtype=np.float64)
    j_sq = j_vals**2
    inv_sqrt2 = 1.0 / np.sqrt(2.0)
    
    # t = 0 observables
    prob = np.abs(psiL)**2 + np.abs(psiR)**2
    m1 = np.sum(j_vals * prob)
    m2 = np.sum(j_sq * prob)
    sigma[0] = np.sqrt(max(0.0, m2 - m1**2))
    
    cL = np.zeros(N_full, dtype=np.complex128)
    cR = np.zeros(N_full, dtype=np.complex128)
    
    for t in range(1, t_max + 1):
        # 1. Hadamard coin in the bulk B = {-a, ..., a}
        for i in range(1, N_full - 1):
            cL[i] = (psiL[i] + psiR[i]) * inv_sqrt2
            cR[i] = (psiL[i] - psiR[i]) * inv_sqrt2
            
        # 2. Pauli-X (NOT) coin at auxiliary reflecting boundaries W = {-(a+1), a+1}
        cL[0] = psiR[0]
        cR[0] = psiL[0]
        cL[N_full - 1] = psiR[N_full - 1]
        cR[N_full - 1] = psiL[N_full - 1]
        
        # 3. Conditional shift S
        for i in range(N_full - 1):
            psiL[i] = cL[i + 1]
        psiL[N_full - 1] = 0.0 + 0.0j
        
        for i in range(N_full - 1, 0, -1):
            psiR[i] = cR[i - 1]
        psiR[0] = 0.0 + 0.0j
        
        # 4. Standard deviation sigma(t)
        prob = np.abs(psiL)**2 + np.abs(psiR)**2
        m1 = np.sum(j_vals * prob)
        m2 = np.sum(j_sq * prob)
        sigma[t] = np.sqrt(max(0.0, m2 - m1**2))
        
    return sigma


def get_initial_states(N=1001):
    """
    Constructs the lattice coordinate array and the three initial states:
    1. Localized: |chi> x |0>
    2. Gaussian:  |chi> x sum_j g(j)|j> with sigma_0 = 3
    3. Uniform:   |chi> x sum_j (1/sqrt(N))|j>
    with symmetric coin state |chi> = (|L> + i|R>) / sqrt(2).
    """
    a = (N - 1) // 2
    j_full = np.arange(-(a + 1), (a + 1) + 1, dtype=np.float64)
    N_full = len(j_full)
    center = a + 1
    
    # 1. Localized initial state
    psiL_loc = np.zeros(N_full, dtype=np.complex128)
    psiR_loc = np.zeros(N_full, dtype=np.complex128)
    psiL_loc[center] = 1.0 / np.sqrt(2.0)
    psiR_loc[center] = 1.0j / np.sqrt(2.0)
    
    # 2. Gaussian initial state (sigma_0 = 3)
    sigma0 = 3.0
    j_bulk = np.arange(-a, a + 1, dtype=np.float64)
    g = np.exp(-j_bulk**2 / (4.0 * sigma0**2))
    g = g / np.sqrt(np.sum(g**2))
    psiL_gauss = np.zeros(N_full, dtype=np.complex128)
    psiR_gauss = np.zeros(N_full, dtype=np.complex128)
    psiL_gauss[1:-1] = (1.0 / np.sqrt(2.0)) * g
    psiR_gauss[1:-1] = (1.0j / np.sqrt(2.0)) * g
    
    # 3. Truncated-uniform initial state
    val = 1.0 / np.sqrt(N)
    psiL_unif = np.zeros(N_full, dtype=np.complex128)
    psiR_unif = np.zeros(N_full, dtype=np.complex128)
    psiL_unif[1:-1] = (1.0 / np.sqrt(2.0)) * val
    psiR_unif[1:-1] = (1.0j / np.sqrt(2.0)) * val
    
    return j_full, (psiL_loc, psiR_loc), (psiL_gauss, psiR_gauss), (psiL_unif, psiR_unif)


def compute_asymptotic_spreading_rates(N=1001):
    r"""
    Computes \bar{\nu}_H for the three initial states:
    - Localized: \sqrt{1 - 1/\sqrt{2}} (Eq. 39)
    - Gaussian (\sigma_0 = 3): 0.7028 (Table I, Ref. [44])
    - Truncated-uniform: \sqrt{I_N} (Eqs. 41-42)
    """
    nu_loc = np.sqrt(1.0 - 1.0 / np.sqrt(2.0))
    nu_gauss = 0.7028
    
    n_idx = np.arange(1, (N - 1) // 2 + 1)
    r = 2.0 * np.sqrt(2.0) - 3.0
    I_N = 1.0 - 1.0 / np.sqrt(2.0) - np.sqrt(2.0) * np.sum((r**n_idx) * (1.0 - 2.0 * n_idx / N))
    nu_unif = np.sqrt(I_N)
    
    return nu_loc, nu_gauss, nu_unif


def load_or_run_simulation(N=1001, t_max=500_000, force_recompute=False):
    """
    Loads precomputed stiffness data from cache or runs the full simulation.
    """
    if os.path.exists(CACHE_FILE) and not force_recompute:
        data = np.load(CACHE_FILE)
        if all(k in data for k in ("S_loc", "S_gauss", "S_unif")) and len(data["S_loc"]) == t_max:
            return data["S_loc"], data["S_gauss"], data["S_unif"]

    print(f"Running simulation for N = {N} up to t = {t_max} steps...")
    j_full, loc_init, gauss_init, unif_init = get_initial_states(N)
    
    sig_loc = _simulate_qw_sigma(loc_init[0], loc_init[1], j_full, t_max)
    sig_gauss = _simulate_qw_sigma(gauss_init[0], gauss_init[1], j_full, t_max)
    sig_unif = _simulate_qw_sigma(unif_init[0], unif_init[1], j_full, t_max)
    
    nu_loc, nu_gauss, nu_unif = compute_asymptotic_spreading_rates(N)
    
    # Spreading rate \nu_\sigma(t) = |\sigma(t) - \sigma(t-1)|
    nu_sig_loc = np.abs(np.diff(sig_loc))
    nu_sig_gauss = np.abs(np.diff(sig_gauss))
    nu_sig_unif = np.abs(np.diff(sig_unif))
    
    # Spreading stiffness S_\nu(t) = \bar{\nu}_H / (\bar{\nu}_H + \nu_\sigma(t))
    S_loc = nu_loc / (nu_loc + nu_sig_loc)
    S_gauss = nu_gauss / (nu_gauss + nu_sig_gauss)
    S_unif = nu_unif / (nu_unif + nu_sig_unif)
    
    np.savez_compressed(CACHE_FILE, S_loc=S_loc, S_gauss=S_gauss, S_unif=S_unif)
    print(f"Data saved to {CACHE_FILE}")
    
    return S_loc, S_gauss, S_unif


def plot_fig3(S_loc, S_gauss, S_unif, t_max=500_000, tc=20_000):
    """
    Generates Fig. 3 with layout, styling, and typography identical to the paper.
    """
    setup_matplotlib_style()
    
    fig, ax = plt.subplots(figsize=(6.8, 4.8), dpi=300)
    
    t_scaled = np.arange(1, t_max + 1) / 1000.0
    
    # Main panel: long-time stiffness evolution
    ax.plot(t_scaled, S_loc, color="black", lw=0.35, alpha=0.65, zorder=2)
    ax.plot(t_scaled, S_gauss, color="red", lw=0.35, alpha=0.65, zorder=3)
    ax.plot(t_scaled, S_unif, color="blue", lw=0.35, alpha=0.65, zorder=4)
    
    # Dashed vertical line marking transient cutoff t_c = 2 x 10^4 (x = 20)
    ax.axvline(x=tc / 1000.0, color="black", linestyle="--", lw=1.2, zorder=5)
    
    ax.set_xlim(0, 500)
    ax.set_ylim(0.0, 1.0)
    ax.set_xticks(np.arange(0, 501, 100))
    ax.set_yticks(np.arange(0.0, 1.1, 0.2))
    ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.1f"))
    
    ax.set_xlabel(r"Time Steps $t$ $(\times 10^3)$", fontsize=12)
    ax.set_ylabel(r"$S_\nu$", fontsize=13)
    
    # Legend
    legend_elements = [
        Line2D([0], [0], color="black", lw=1.5, label="Localized"),
        Line2D([0], [0], color="red", lw=1.5, label=r"Gaussian ($\sigma_0 = 3$)"),
        Line2D([0], [0], color="blue", lw=1.5, label="Truncated uniform"),
    ]
    ax.legend(
        handles=legend_elements,
        loc=(0.14, 0.38),
        frameon=False,
        fontsize=11
    )
    
    # Inset: initial dynamics over t <= t_c
    ax_ins = ax.inset_axes([0.56, 0.14, 0.40, 0.46])
    ax_ins.set_facecolor("white")
    
    t_inset = np.arange(1, tc + 1) / 1000.0
    ax_ins.plot(t_inset, S_loc[:tc], color="black", lw=0.8, zorder=2)
    ax_ins.plot(t_inset, S_gauss[:tc], color="red", lw=0.8, zorder=3)
    ax_ins.plot(t_inset, S_unif[:tc], color="blue", lw=0.8, zorder=4)
    
    ax_ins.set_xlim(0, 20)
    ax_ins.set_ylim(0.5, 1.0)
    ax_ins.set_xticks([0, 5, 10, 15, 20])
    ax_ins.set_yticks([0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    ax_ins.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.1f"))
    ax_ins.set_ylabel(r"$S_\nu(t)$", fontsize=11)
    ax_ins.tick_params(axis="both", which="major", labelsize=10, direction="in", top=True, right=True)
    
    fig.subplots_adjust(left=0.10, right=0.96, top=0.96, bottom=0.12)
    
    pdf_out = "Figura03.pdf"
    #png_out = "Figura03.png"
    plt.savefig(pdf_out)
    #plt.savefig(png_out, dpi=300)
    #print(f"Saved figure to {pdf_out} and {png_out}")
    plt.close()


def main():
    S_loc, S_gauss, S_unif = load_or_run_simulation(N=1001, t_max=500_000)
    plot_fig3(S_loc, S_gauss, S_unif, t_max=500_000, tc=20_000)


if __name__ == "__main__":
    main()
