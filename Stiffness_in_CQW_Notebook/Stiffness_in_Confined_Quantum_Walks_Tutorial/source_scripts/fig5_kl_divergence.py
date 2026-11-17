"""
FIG. 5
'Stiffness in Confined Quantum Walks'
Pablo G. M. Dantas, Rafael Vieira, Edgard P. M. Amorim

Normalized Kullback-Leibler divergence for localized, Gaussian, and
truncated-uniform initial states. The localized state uses the two-step
parity-corrected distribution described in the paper.
"""

import os
import time
from concurrent.futures import ThreadPoolExecutor

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from numba import njit

from qw_simulation import get_initial_states, setup_matplotlib_style

CODE_DIR = os.path.dirname(os.path.abspath(__file__))
IMAGE_DIR = os.path.join(os.path.dirname(CODE_DIR), "images")
CACHE_FILE = os.path.join(CODE_DIR, "fig5_cache.npz")


@njit(fastmath=True, nogil=True)
def _simulate_dkl(psiL_init, psiR_init, j_vals, t_max, tc, record_series):
    """Evolve all three walks and compute normalized KL divergence."""
    n_states, n_sites = psiL_init.shape
    psiL = psiL_init.copy()
    psiR = psiR_init.copy()
    cL = np.zeros((n_states, n_sites), dtype=np.complex128)
    cR = np.zeros((n_states, n_sites), dtype=np.complex128)
    p_prev = np.zeros((n_states, n_sites), dtype=np.float64)
    p_curr = np.zeros((n_states, n_sites), dtype=np.float64)
    if record_series:
        dkl_series = np.zeros((n_states, t_max + 1), dtype=np.float64)
    else:
        dkl_series = np.zeros((n_states, 0), dtype=np.float64)
    dkl_values = np.zeros(n_states, dtype=np.float64)
    moments = np.zeros((n_states, 2), dtype=np.float64)
    inv_sqrt2 = 1.0 / np.sqrt(2.0)

    for state in range(n_states):
        bulk_total = 0.0
        for i in range(1, n_sites - 1):
            probability = abs(psiL[state, i]) ** 2 + abs(psiR[state, i]) ** 2
            p_prev[state, i] = probability
            bulk_total += probability
        for i in range(1, n_sites - 1):
            p_prev[state, i] /= bulk_total

    # The localized two-step divergence is zero at t=0.
    dkl_values[0] = 0.0
    for state in range(1, n_states):
        entropy = 0.0
        for i in range(1, n_sites - 1):
            probability = p_prev[state, i]
            if probability > 0.0:
                entropy -= probability * np.log(probability)
        dkl_values[state] = 1.0 - entropy / np.log(n_sites - 2.0)
    if record_series:
        for state in range(n_states):
            dkl_series[state, 0] = dkl_values[state]

    a = (n_sites - 3) // 2
    m_prev = 1

    for t in range(1, t_max + 1):
        for state in range(n_states):
            for i in range(1, n_sites - 1):
                cL[state, i] = (psiL[state, i] + psiR[state, i]) * inv_sqrt2
                cR[state, i] = (psiL[state, i] - psiR[state, i]) * inv_sqrt2
            cL[state, 0] = psiR[state, 0]
            cR[state, 0] = psiL[state, 0]
            cL[state, n_sites - 1] = psiR[state, n_sites - 1]
            cR[state, n_sites - 1] = psiL[state, n_sites - 1]

            for i in range(n_sites - 1):
                psiL[state, i] = cL[state, i + 1]
            psiL[state, n_sites - 1] = 0.0 + 0.0j
            for i in range(n_sites - 1, 0, -1):
                psiR[state, i] = cR[state, i - 1]
            psiR[state, 0] = 0.0 + 0.0j

            bulk_total = 0.0
            for i in range(1, n_sites - 1):
                probability = abs(psiL[state, i]) ** 2 + abs(psiR[state, i]) ** 2
                p_curr[state, i] = probability
                bulk_total += probability
            for i in range(1, n_sites - 1):
                p_curr[state, i] /= bulk_total

        k = min(t, a)
        m_curr = k + 1 if k % 2 == t % 2 else k
        m_two_step = m_prev + m_curr
        entropy = 0.0
        for i in range(1, n_sites - 1):
            probability = (
                m_prev * p_prev[0, i] + m_curr * p_curr[0, i]
            ) / m_two_step
            if probability > 0.0:
                entropy -= probability * np.log(probability)
        dkl_values[0] = 1.0 - entropy / np.log(m_two_step)

        for state in range(1, n_states):
            entropy = 0.0
            for i in range(1, n_sites - 1):
                probability = p_curr[state, i]
                if probability > 0.0:
                    entropy -= probability * np.log(probability)
            dkl_values[state] = 1.0 - entropy / np.log(n_sites - 2.0)

        if record_series:
            for state in range(n_states):
                dkl_series[state, t] = dkl_values[state]
        if t > tc:
            for state in range(n_states):
                value = dkl_values[state]
                moments[state, 0] += value
                moments[state, 1] += value * value

        p_prev, p_curr = p_curr, p_prev
        m_prev = m_curr

    if t_max > tc:
        moments /= t_max - tc
    return dkl_series, moments


def _initial_state_arrays(N):
    a = (N - 1) // 2
    j_full, localized, gaussian, uniform = get_initial_states(a)
    psiL = np.stack((localized[0], gaussian[0], uniform[0]))
    psiR = np.stack((localized[1], gaussian[1], uniform[1]))
    return j_full, psiL, psiR


def _compute_finite_size_point(N, t_max, tc):
    j_full, psiL, psiR = _initial_state_arrays(N)
    _, moments = _simulate_dkl(
        psiL, psiR, j_full, t_max, tc, False
    )
    return N, moments[:, 0], moments[:, 1]


def load_or_run_simulation(
    t_max=500_000,
    tc=100_000,
    N_values=None,
    force_recompute=False,
):
    """Load cached statistics or calculate the time series and finite-size data."""
    if t_max <= tc:
        raise ValueError("t_max must be greater than tc.")
    if N_values is None:
        N_values = np.arange(101, 1002, 10, dtype=np.int64)
    else:
        N_values = np.asarray(N_values, dtype=np.int64)
    if (
        N_values.ndim != 1
        or N_values.size == 0
        or np.any(N_values < 3)
        or np.any(N_values % 2 == 0)
    ):
        raise ValueError("N_values must contain positive odd lattice sizes >= 3.")

    cached = {}
    if os.path.exists(CACHE_FILE) and not force_recompute:
        with np.load(CACHE_FILE) as saved:
            matches = (
                "t_max" in saved
                and "tc" in saved
                and "N_values" in saved
                and int(saved["t_max"]) == t_max
                and int(saved["tc"]) == tc
                and np.array_equal(saved["N_values"], N_values)
            )
            if matches:
                cached = {key: saved[key] for key in saved.files}

    if "dkl_series" not in cached or cached["dkl_series"].shape != (3, t_max + 1):
        print(f"Running Figure 5 time-series simulation for N = 1001 to t = {t_max}...")
        j_full, psiL, psiR = _initial_state_arrays(1001)
        dkl_series, _ = _simulate_dkl(psiL, psiR, j_full, t_max, tc, True)
        cached.update(
            t_max=np.array(t_max),
            tc=np.array(tc),
            N_values=N_values,
            dkl_series=dkl_series,
        )
        np.savez_compressed(CACHE_FILE, **cached)

    if (
        "mean_dkl" not in cached
        or "mean_dkl2" not in cached
        or cached.get("mean_dkl", np.empty(0)).shape != (3, N_values.size)
        or cached.get("mean_dkl2", np.empty(0)).shape != (3, N_values.size)
        or cached.get("N_arr", np.empty(0)).shape != N_values.shape
        or not np.array_equal(cached.get("N_arr", np.empty(0)), N_values)
    ):
        j_full, psiL, psiR = _initial_state_arrays(11)
        _simulate_dkl(psiL, psiR, j_full, 10, 5, False)
        print(f"Running finite-size simulations for {len(N_values)} confinement sizes...")
        start_time = time.time()
        max_workers = min(12, os.cpu_count() or 4)
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            results = list(
                executor.map(
                    lambda N: _compute_finite_size_point(N, t_max, tc),
                    N_values,
                )
            )
        results.sort(key=lambda result: result[0])
        cached["N_arr"] = np.array([result[0] for result in results], dtype=np.int64)
        cached["mean_dkl"] = np.stack(
            [result[1] for result in results], axis=1
        )
        cached["mean_dkl2"] = np.stack(
            [result[2] for result in results], axis=1
        )
        print(f"Finite-size simulations completed in {time.time() - start_time:.2f} s.")
        np.savez_compressed(CACHE_FILE, **cached)

    return cached


def plot_fig5(data, pdf_out=None):
    """Generate Figure 5 with its time-series and finite-size panels."""
    if pdf_out is None:
        pdf_out = os.path.join(IMAGE_DIR, "Figura05.pdf")
    setup_matplotlib_style()
    colors = ("black", "red", "blue")
    labels = (
        "Localized",
        r"Gaussian ($\sigma_0 = 3$)",
        "Truncated uniform",
    )

    fig, (ax_time, ax_size) = plt.subplots(
        2, 1, figsize=(6.8, 7.2), dpi=300
    )

    t_max = len(data["dkl_series"][0]) - 1
    t_scaled = np.arange(t_max + 1) / 1000.0
    for state, color in enumerate(colors):
        ax_time.plot(
            t_scaled,
            data["dkl_series"][state],
            color=color,
            lw=0.35,
            alpha=0.65,
        )
    ax_time.set_xlim(0, t_max / 1000.0)
    ax_time.set_ylim(0, 0.5)
    ax_time.set_xticks(np.arange(0, t_max / 1000.0 + 1, 100))
    ax_time.set_yticks(np.arange(0, 0.51, 0.1))
    ax_time.set_xlabel(r"Time Steps $t$ $(\times 10^3)$", fontsize=12)
    ax_time.set_ylabel(r"$D_{\mathrm{KL}}$", fontsize=13)
    ax_time.text(0.05, 0.90, "(a)", transform=ax_time.transAxes, fontsize=12)

    ax_time.axvline(
        x=data["tc"] / 1000.0,
        color="black",
        linestyle="--",
        lw=1.2,
        zorder=6,
    )

    ax_early = ax_time.inset_axes([0.53, 0.16, 0.42, 0.46])
    early_end = min(2000, t_max)
    early_t = np.arange(early_end + 1) / 1000.0
    for state, color in enumerate(colors):
        ax_early.plot(
            early_t,
            data["dkl_series"][state, : early_end + 1],
            color=color,
            lw=0.8,
        )
    ax_early.set_xlim(0, 2)
    ax_early.set_ylim(-0.05, 0.6)
    ax_early.set_xticks([0, 0.5, 1.0, 1.5, 2.0])
    ax_early.set_yticks(np.arange(0, 0.61, 0.1))
    ax_early.set_ylabel(r"$D_{\mathrm{KL}}$", fontsize=11)
    ax_early.tick_params(axis="both", which="major", labelsize=9, direction="in", top=True, right=True)

    legend_handles = [
        Line2D([0], [0], color=color, lw=1.5, label=label)
        for color, label in zip(colors, labels)
    ]
    fig.legend(
        handles=legend_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.99),
        ncol=3,
        frameon=False,
        fontsize=11,
    )

    N_arr = data["N_arr"]
    mean_dkl = data["mean_dkl"]
    mean_dkl2 = data["mean_dkl2"]
    for state, color in enumerate(colors):
        ax_size.plot(N_arr, mean_dkl[state], color=color, lw=1.5)
    ax_size.set_xlim(100, 1000)
    ax_size.set_ylim(0, 0.15)
    ax_size.set_xticks([200, 400, 600, 800, 1000])
    ax_size.set_yticks([0, 0.05, 0.10, 0.15])
    ax_size.set_xlabel(r"$N$", fontsize=13)
    ax_size.set_ylabel(r"$\langle D_{\mathrm{KL}} \rangle$", fontsize=13)
    ax_size.text(0.05, 0.90, "(b)", transform=ax_size.transAxes, fontsize=12)

    ax_moment = ax_size.inset_axes([0.54, 0.43, 0.40, 0.47])
    for state, color in enumerate(colors):
        ax_moment.plot(N_arr, mean_dkl2[state], color=color, lw=1.2)
    ax_moment.set_xlim(100, 1000)
    ax_moment.set_ylim(0, 0.003)
    ax_moment.set_xticks([200, 400, 600, 800, 1000])
    ax_moment.set_yticks(np.arange(0, 0.0031, 0.0005))
    ax_moment.set_ylabel(r"$\langle D_{\mathrm{KL}}^2 \rangle$", fontsize=10)
    ax_moment.tick_params(axis="both", which="major", labelsize=8, direction="in", top=True, right=True)

    fig.subplots_adjust(left=0.12, right=0.96, top=0.92, bottom=0.09, hspace=0.30)
    fig.savefig(pdf_out)
    plt.close(fig)


def main():
    data = load_or_run_simulation()
    plot_fig5(data)


if __name__ == "__main__":
    main()
