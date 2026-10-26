r"""
Reprodução do Painel (j) da FIG. 2
'Stiffness in Confined Quantum Walks'
Evolução inicial da coerência de moeda C(t) = |\rho_{LR}(t)|^2 para t <= 1000 passos.
"""

import numpy as np
import matplotlib.pyplot as plt
from qw_simulation import load_or_run_simulation, setup_matplotlib_style

def main():
    setup_matplotlib_style()
    data = load_or_run_simulation()
    
    t_steps = 1000
    t_axis = np.arange(t_steps + 1) / 1000.0  # Em unidades de 10^3 passos
    
    c_loc = data['c_loc'][:t_steps + 1]
    c_gauss = data['c_gauss'][:t_steps + 1]
    c_unif = data['c_unif'][:t_steps + 1]
    
    fig, ax = plt.subplots(figsize=(6, 4.2), dpi=200)
    
    # Curvas de coerência de moeda
    ax.plot(t_axis, c_unif, color='blue', lw=1.5, label='Truncated Uniform state', zorder=2)
    ax.plot(t_axis, c_gauss, color='red', lw=1.5, label=r'Gaussian state ($\sigma_0 = 3$)', zorder=3)
    ax.plot(t_axis, c_loc, color='black', lw=1.0, label='Localized', zorder=4)
    
    # Rótulo do painel
    ax.text(0.04, 0.90, "(j)", transform=ax.transAxes, fontsize=13)
    
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.00, 0.30)
    ax.set_xticks(np.arange(0.0, 1.1, 0.2))
    ax.set_yticks(np.arange(0.00, 0.31, 0.05))
    ax.yaxis.set_major_formatter(plt.FormatStrFormatter('%.2f'))
    
    ax.set_xlabel(r"Time Steps $t \, (\times 10^3)$", fontsize=12)
    ax.set_ylabel(r"$C$", fontsize=13)
    ax.legend(frameon=False, loc="upper right", fontsize=9)
    
    plt.tight_layout()
    plt.savefig("fig2_panel_j.png", dpi=300)
    print("Salvo: fig2_panel_j.png")
    plt.close()

if __name__ == "__main__":
    main()

