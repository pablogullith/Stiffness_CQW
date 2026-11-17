"""
Confined Discrete-Time Quantum Walk Simulation Module
Based on the paper:
'Stiffness in Confined Quantum Walks'
Authors: Pablo G. M. Dantas, Rafael Vieira, Edgard P. M. Amorim
"""

import os
import numpy as np
from numba import njit

CACHE_FILE = "qw_simulation_cache.npz"

@njit(fastmath=True)
def _simulate_qw_core(psiL_init, psiR_init, j_vals, t_max, record_snapshots, snapshot_times):
    """
    Numba-optimized core for unitary evolution of the confined quantum walk.
    Lattice: j in [-a-1, a+1] with bulk in [-a, a] and NOT boundaries at +/- (a+1).
    """
    N_full = len(psiL_init)
    psiL = psiL_init.copy()
    psiR = psiR_init.copy()
    
    sigma = np.zeros(t_max + 1, dtype=np.float64)
    C = np.zeros(t_max + 1, dtype=np.float64)
    
    num_snaps = len(snapshot_times) if record_snapshots else 0
    snap_probs = np.zeros((num_snaps, N_full), dtype=np.float64)
    snap_idx = 0
    
    inv_sqrt2 = 1.0 / np.sqrt(2.0)
    
    # t = 0 observables
    prob = np.abs(psiL)**2 + np.abs(psiR)**2
    m1 = np.sum(j_vals * prob)
    m2 = np.sum(j_vals**2 * prob)
    sigma[0] = np.sqrt(max(0.0, m2 - m1**2))
    rho_LR = np.sum(psiL * np.conj(psiR))
    C[0] = np.abs(rho_LR)**2
    
    cL = np.zeros(N_full, dtype=np.complex128)
    cR = np.zeros(N_full, dtype=np.complex128)
    
    for t in range(1, t_max + 1):
        # 1. Coin in the bulk (Hadamard)
        for i in range(1, N_full - 1):
            cL[i] = (psiL[i] + psiR[i]) * inv_sqrt2
            cR[i] = (psiL[i] - psiR[i]) * inv_sqrt2
            
        # 2. Coin at the boundaries (NOT / Pauli-X)
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
        
        # 4. Observables
        prob = np.abs(psiL)**2 + np.abs(psiR)**2
        m1 = np.sum(j_vals * prob)
        m2 = np.sum(j_vals**2 * prob)
        sigma[t] = np.sqrt(max(0.0, m2 - m1**2))
        
        rho_LR = np.sum(psiL * np.conj(psiR))
        C[t] = np.abs(rho_LR)**2
        
        # Record snapshots
        if record_snapshots and snap_idx < num_snaps:
            if t == snapshot_times[snap_idx]:
                for k in range(N_full):
                    snap_probs[snap_idx, k] = prob[k]
                snap_idx += 1
                
    return sigma, C, snap_probs


def get_initial_states(a=50):
    """
    Returns the lattice j and the initial states:
    1. Localized: |chi> x |0>
    2. Gaussian:  |chi> x sum_j g(j)|j> with sigma_0 = 3
    3. Uniform:   |chi> x sum_j (1/sqrt(N))|j> with N = 101
    with |chi> = (|L> + i|R>) / sqrt(2).
    """
    j_full = np.arange(-(a + 1), (a + 1) + 1, dtype=np.float64)
    N_full = len(j_full)
    center = a + 1
    
    # 1. Localized
    psiL_loc = np.zeros(N_full, dtype=np.complex128)
    psiR_loc = np.zeros(N_full, dtype=np.complex128)
    psiL_loc[center] = 1.0 / np.sqrt(2.0)
    psiR_loc[center] = 1.0j / np.sqrt(2.0)
    
    # 2. Gaussian (sigma_0 = 3)
    sigma0 = 3.0
    j_bulk = np.arange(-a, a + 1, dtype=np.float64)
    g = np.exp(-j_bulk**2 / (4.0 * sigma0**2))
    g = g / np.sqrt(np.sum(g**2))
    psiL_gauss = np.zeros(N_full, dtype=np.complex128)
    psiR_gauss = np.zeros(N_full, dtype=np.complex128)
    psiL_gauss[1:-1] = (1.0 / np.sqrt(2.0)) * g
    psiR_gauss[1:-1] = (1.0j / np.sqrt(2.0)) * g
    
    # 3. Truncated uniform
    val = 1.0 / np.sqrt(2 * a + 1)
    psiL_unif = np.zeros(N_full, dtype=np.complex128)
    psiR_unif = np.zeros(N_full, dtype=np.complex128)
    psiL_unif[1:-1] = (1.0 / np.sqrt(2.0)) * val
    psiR_unif[1:-1] = (1.0j / np.sqrt(2.0)) * val
    
    return j_full, (psiL_loc, psiR_loc), (psiL_gauss, psiR_gauss), (psiL_unif, psiR_unif)


def load_or_run_simulation(force_recompute=False):
    """
    Runs the full simulation up to 500,000 steps for the 3 states
    and saves/loads it in a compressed cache file.
    """
    t_snaps_loc = np.array([41, 82, 122, 163, 1630, 16300, 500000], dtype=np.int64)
    t_snaps_gauss = np.array([36, 73, 110, 146, 1460, 14600, 500000], dtype=np.int64)
    t_snaps_unif = np.array([36, 72, 108, 144, 1440, 14400, 500000], dtype=np.int64)

    if os.path.exists(CACHE_FILE) and not force_recompute:
        with np.load(CACHE_FILE) as cached:
            expected_times = {
                't_snaps_loc': t_snaps_loc,
                't_snaps_gauss': t_snaps_gauss,
                't_snaps_unif': t_snaps_unif,
            }
            cache_matches = all(
                key in cached.files and np.array_equal(cached[key], times)
                for key, times in expected_times.items()
            )
            cache_matches = cache_matches and all(
                key in cached.files and cached[key].shape[0] == len(times)
                for key, times in (
                    ('snaps_loc', t_snaps_loc),
                    ('snaps_gauss', t_snaps_gauss),
                    ('snaps_unif', t_snaps_unif),
                )
            )
            if cache_matches:
                return {k: cached[k] for k in cached.files}
        print("Cache outdated; regenerating snapshots.")
        
    print("Generating simulation data (500,000 steps)...")
    a = 50
    j_full, loc_init, gauss_init, unif_init = get_initial_states(a)
    
    t_max = 500_000
    
    s_loc, c_loc, snaps_loc = _simulate_qw_core(loc_init[0], loc_init[1], j_full, t_max, True, t_snaps_loc)
    s_gauss, c_gauss, snaps_gauss = _simulate_qw_core(gauss_init[0], gauss_init[1], j_full, t_max, True, t_snaps_gauss)
    s_unif, c_unif, snaps_unif = _simulate_qw_core(unif_init[0], unif_init[1], j_full, t_max, True, t_snaps_unif)
    
    results = {
        'j_full': j_full,
        's_loc': s_loc,
        'c_loc': c_loc,
        'snaps_loc': snaps_loc,
        't_snaps_loc': t_snaps_loc,
        's_gauss': s_gauss,
        'c_gauss': c_gauss,
        'snaps_gauss': snaps_gauss,
        't_snaps_gauss': t_snaps_gauss,
        's_unif': s_unif,
        'c_unif': c_unif,
        'snaps_unif': snaps_unif,
        't_snaps_unif': t_snaps_unif,
    }
    
    np.savez_compressed(CACHE_FILE, **results)
    print("Data saved to", CACHE_FILE)
    return results


def setup_matplotlib_style():
    """Configures the style identical to the paper (Computer Modern / STIX serif)."""
    import matplotlib.pyplot as plt
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