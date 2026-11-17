r"""
FIG. 4
'Stiffness in Confined Quantum Walks'
Pablo G. M. Dantas, Rafael Vieira, Edgard P. M. Amorim

Finite-size dependence of the long-time averaged spreading stiffness \langle S_\nu \rangle
for the localized (black), Gaussian with \sigma_0 = 3 (red), and truncated-uniform (blue) initial states.
The localized and truncated-uniform states exhibit an overall increase of the mean stiffness
with the confinement size N, whereas the Gaussian state reaches a broad maximum at intermediate
sizes and subsequently decreases.
The inset shows 1 - \langle S_\nu \rangle as a function of 1 - \langle S_\nu^2 \rangle for the same
three initial states, distinguished by line style where their curves overlap. The dashed line
represents the high-stiffness reference relation
1 - \langle S_\nu \rangle = (1/2) [1 - \langle S_\nu^2 \rangle].
For the finite-size analysis, the first 2 x 10^4 time steps are discarded and the averages
are evaluated over the subsequent 4.8 x 10^5 steps.
"""

import os
import time
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import matplotlib.ticker as ticker
from numba import njit

CACHE_FILE = "fig4_cache.npz"


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


@njit(fastmath=True, nogil=True)
def _simulate_qw_moments(psiL_init, psiR_init, j_vals, nu_H, t_max, tc):
    """
    Numba-optimized simulation to compute the 1st and 2nd stiffness moments
    without allocating large time-series arrays.
    Lattice: j in [-(a+1), a+1] with Hadamard bulk in [-a, a] and Pauli-X boundaries at +/- (a+1).
    Averages are evaluated over t in [tc + 1, t_max].
    """
    N_full = len(psiL_init)
    psiL = psiL_init.copy()
    psiR = psiR_init.copy()
    j_sq = j_vals**2
    inv_sqrt2 = 1.0 / np.sqrt(2.0)
    
    cL = np.zeros(N_full, dtype=np.complex128)
    cR = np.zeros(N_full, dtype=np.complex128)
    
    # t = 0
    prob = np.abs(psiL)**2 + np.abs(psiR)**2
    m1 = np.sum(j_vals * prob)
    m2 = np.sum(j_sq * prob)
    sigma_prev = np.sqrt(max(0.0, m2 - m1**2))
    
    sum_S = 0.0
    sum_S2 = 0.0
    
    for t in range(1, t_max + 1):
        # 1. Hadamard coin in the bulk
        for i in range(1, N_full - 1):
            cL[i] = (psiL[i] + psiR[i]) * inv_sqrt2
            cR[i] = (psiL[i] - psiR[i]) * inv_sqrt2
            
        # 2. Pauli-X (NOT) coin at boundaries
        cL[0] = psiR[0]
        cR[0] = psiL[0]
        cL[N_full - 1] = psiR[N_full - 1]
        cR[N_full - 1] = psiL[N_full - 1]
        
        # 3. Conditional shift
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
        sigma_curr = np.sqrt(max(0.0, m2 - m1**2))
        
        # 5. Stiffness accumulation for t > tc
        if t > tc:
            nu_sig = abs(sigma_curr - sigma_prev)
            S = nu_H / (nu_H + nu_sig)
            sum_S += S
            sum_S2 += S * S
            
        sigma_prev = sigma_curr
        
    Tav = t_max - tc
    return sum_S / Tav, sum_S2 / Tav


def _compute_single_N(N, t_max=500_000, tc=20_000):
    """Computes stiffness moments for a given confinement size N."""
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

    # Asymptotic free Hadamard velocities \bar{\nu}_H
    nu_loc = np.sqrt(1.0 - 1.0 / np.sqrt(2.0))
    nu_gauss = 0.7028
    n_idx = np.arange(1, (N - 1) // 2 + 1)
    r = 2.0 * np.sqrt(2.0) - 3.0
    I_N = 1.0 - 1.0 / np.sqrt(2.0) - np.sqrt(2.0) * np.sum((r**n_idx) * (1.0 - 2.0 * n_idx / N))
    nu_unif = np.sqrt(I_N)

    sl, s2l = _simulate_qw_moments(psiL_loc, psiR_loc, j_full, nu_loc, t_max, tc)
    sg, s2g = _simulate_qw_moments(psiL_gauss, psiR_gauss, j_full, nu_gauss, t_max, tc)
    su, s2u = _simulate_qw_moments(psiL_unif, psiR_unif, j_full, nu_unif, t_max, tc)
    
    return N, sl, s2l, sg, s2g, su, s2u


def load_or_run_finite_size_simulation(force_recompute=False, t_max=500_000, tc=20_000):
    """
    Loads precomputed finite-size stiffness moments or runs the multithreaded simulation.
    """
    if os.path.exists(CACHE_FILE) and not force_recompute:
        data = np.load(CACHE_FILE)
        required = ("N_arr", "S_loc_m1", "S_loc_m2", "S_gauss_m1", "S_gauss_m2", "S_unif_m1", "S_unif_m2")
        if all(k in data for k in required):
            return {k: data[k] for k in required}

    # Warm-up JIT
    _compute_single_N(11, t_max=10, tc=5)

    N_vals = np.arange(101, 1002, 10)
    print(f"Running multithreaded simulation for {len(N_vals)} values of N from {N_vals[0]} to {N_vals[-1]}...")
    t0 = time.time()
    
    # ThreadPoolExecutor leverages multi-core CPU with nogil in Numba
    max_workers = min(12, os.cpu_count() or 4)
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        results = list(executor.map(lambda n: _compute_single_N(n, t_max=t_max, tc=tc), N_vals))

    results.sort(key=lambda x: x[0])
    
    res_dict = {
        "N_arr": np.array([r[0] for r in results]),
        "S_loc_m1": np.array([r[1] for r in results]),
        "S_loc_m2": np.array([r[2] for r in results]),
        "S_gauss_m1": np.array([r[3] for r in results]),
        "S_gauss_m2": np.array([r[4] for r in results]),
        "S_unif_m1": np.array([r[5] for r in results]),
        "S_unif_m2": np.array([r[6] for r in results]),
    }
    
    np.savez_compressed(CACHE_FILE, **res_dict)
    print(f"Completed in {time.time() - t0:.2f} s. Data saved to {CACHE_FILE}")
    
    return res_dict


def plot_fig4(data):
    """
    Generates Fig. 4 with layout, proportions, and typography identical to the paper.
    """
    setup_matplotlib_style()
    
    N_arr = data["N_arr"]
    S_loc_m1 = data["S_loc_m1"]
    S_loc_m2 = data["S_loc_m2"]
    S_gauss_m1 = data["S_gauss_m1"]
    S_gauss_m2 = data["S_gauss_m2"]
    S_unif_m1 = data["S_unif_m1"]
    S_unif_m2 = data["S_unif_m2"]

    col_loc = "black"
    col_gauss = "red"
    col_unif = "blue"
    style_loc = "-"
    style_gauss = "--"
    style_unif = ":"

    fig, ax = plt.subplots(figsize=(6.8, 5.0), dpi=300)

    
    # Main Panel: <S_\nu> vs N
    ax.plot(N_arr, S_unif_m1, color=col_unif, lw=1.5, zorder=4)
    ax.plot(N_arr, S_loc_m1, color=col_loc, lw=1.5, zorder=3)
    ax.plot(N_arr, S_gauss_m1, color=col_gauss, lw=1.5, zorder=2)

    ax.set_xlim(100, 1000)
    ax.set_ylim(0.48, 1.01)
    ax.set_xticks([200, 400, 600, 800, 1000])
    ax.set_yticks([0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.1f"))

    ax.set_xlabel(r"$N$", fontsize=13)
    ax.set_ylabel(r"$\langle S_\nu \rangle$", fontsize=13)

    
    # Inset: 1 - <S_\nu> vs 1 - <S_\nu^2>
    ax_ins = ax.inset_axes([0.15, 0.14, 0.44, 0.48])
    ax_ins.set_facecolor("white")

    x_unif = 1.0 - S_unif_m2
    y_unif = 1.0 - S_unif_m1

    x_loc = 1.0 - S_loc_m2
    y_loc = 1.0 - S_loc_m1

    x_gauss = 1.0 - S_gauss_m2
    y_gauss = 1.0 - S_gauss_m1

    # Asymptotic consistency relation reference line: 1 - <S> = (1/2) [1 - <S^2>]
    x_ref = np.linspace(0.0, 0.25, 100)
    y_ref = 0.5 * x_ref
    ax_ins.plot(x_ref, y_ref, color="gray", linestyle="--", lw=1.2, zorder=1)

    ax_ins.plot(x_loc, y_loc, color=col_loc, linestyle=style_loc, lw=1.4, zorder=2)
    ax_ins.plot(x_gauss, y_gauss, color=col_gauss, linestyle=style_gauss, lw=1.4, zorder=3)
    ax_ins.plot(x_unif, y_unif, color=col_unif, linestyle=style_unif, lw=1.4, zorder=4)

    ax_ins.set_xlim(0.00, 0.25)
    ax_ins.set_ylim(0.00, 0.14)
    ax_ins.set_xticks([0.00, 0.05, 0.10, 0.15, 0.20, 0.25])
    ax_ins.set_yticks([0.00, 0.02, 0.04, 0.06, 0.08, 0.10, 0.12, 0.14])
    ax_ins.xaxis.set_major_formatter(ticker.FormatStrFormatter("%.2f"))
    ax_ins.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.2f"))

    ax_ins.set_xlabel(r"$1 - \langle S_\nu^2 \rangle$", fontsize=10)
    ax_ins.set_ylabel(r"$1 - \langle S_\nu \rangle$", fontsize=10)
    ax_ins.tick_params(axis="both", which="major", labelsize=8.5, direction="in", top=True, right=True)

    
    # Legend: moved outside the inset subplot, to the right of it
    legend_elements = [
        Line2D([0], [0], color=col_loc, lw=1.5, label="Localized"),
        Line2D([0], [0], color=col_gauss, lw=1.5, label=r"Gaussian ($\sigma_0 = 3$)"),
        Line2D([0], [0], color=col_unif, lw=1.5, label="Truncated uniform"),
    ]
    ax.legend(
        handles=legend_elements,
        loc="center left",
        bbox_to_anchor=(0.63, 0.35),
        frameon=False,
        fontsize=11,
        handlelength=1.8,
        labelspacing=0.5
    )

    fig.subplots_adjust(left=0.12, right=0.96, top=0.96, bottom=0.12)
    
    pdf_out = "Figura04.pdf"
    #png_out = "Figura04.png"
    plt.savefig(pdf_out)
    #plt.savefig(png_out, dpi=300)
    #print(f"Saved figure to {pdf_out} and {png_out}")
    plt.close()


def main():
    data = load_or_run_finite_size_simulation()
    plot_fig4(data)


if __name__ == "__main__":
    main()
