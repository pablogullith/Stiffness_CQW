r"""
Reprodução do Painel (g) da FIG. 2
'Stiffness in Confined Quantum Walks'
Evolução inicial do desvio padrão \sigma(t) para t <= 1000 passos (t in [0.0, 1.0] x 10^3).
"""

import numpy as np
import matplotlib.pyplot as plt
from qw_simulation import load_or_run_simulation, setup_matplotlib_style

def main():
    setup_matplotlib_style()
    data = load_or_run_simulation()
    
    t_steps = 1000
    t_axis = np.arange(t_steps + 1) / 1000.0  # Em unidades de 10^3 passos
    
    s_loc = data['s_loc'][:t_steps + 1]
    s_gauss = data['s_gauss'][:t_steps + 1]
    s_unif = data['s_unif'][:t_steps + 1]
    
    fig, ax = plt.subplots(figsize=(6, 4.2), dpi=200)
    
    # Curvas dos três estados
    ax.plot(t_axis, s_unif, color='blue', lw=1.5, label='Truncated Uniform state', zorder=2)
    ax.plot(t_axis, s_gauss, color='red', lw=1.5, label=r'Gaussian state ($\sigma_0 = 3$)', zorder=3)
    ax.plot(t_axis, s_loc, color='black', lw=1.3, label='Localized', zorder=4)
    
    # Rótulo do painel
    ax.text(0.04, 0.90, "(g)", transform=ax.transAxes, fontsize=13)
    
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0, 60)
    ax.set_xticks(np.arange(0.0, 1.1, 0.2))
    ax.set_yticks(np.arange(0, 61, 10))
    
    ax.set_xlabel(r"Time Steps $t \, (\times 10^3)$", fontsize=12)
    ax.set_ylabel(r"$\sigma$", fontsize=13)
    ax.legend(frameon=False, loc="upper right", fontsize=9)
    
    plt.tight_layout()
    plt.savefig("fig2_panel_g.png", dpi=300)
    print("Salvo: fig2_panel_g.png")
    plt.close()

if __name__ == "__main__":
    main()

