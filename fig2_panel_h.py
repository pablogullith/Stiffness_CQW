r"""
Reprodução do Painel (h) da FIG. 2
'Stiffness in Confined Quantum Walks'
Evolução em tempos longos do desvio padrão \sigma(t) até 5 x 10^5 passos (t in [0, 500] x 10^3).
"""

import numpy as np
import matplotlib.pyplot as plt
from qw_simulation import load_or_run_simulation, setup_matplotlib_style

def main():
    setup_matplotlib_style()
    data = load_or_run_simulation()
    
    t_max = 500_000
    t_axis = np.arange(t_max + 1) / 1000.0  # Em unidades de 10^3 passos
    
    s_loc = data['s_loc'][:t_max + 1]
    s_gauss = data['s_gauss'][:t_max + 1]
    s_unif = data['s_unif'][:t_max + 1]
    
    fig, ax = plt.subplots(figsize=(6, 4.2), dpi=200)
    
    # Devido à alta densidade de pontos (500.000), o traçado com alpha reproduz fielmente
    # a banda de oscilação e modulação lenta observada no artigo
    ax.plot(t_axis, s_unif, color='#5c82ff', alpha=0.55, lw=0.4, label='Truncated Uniform state', zorder=2)
    ax.plot(t_axis, s_gauss, color='#ff5050', alpha=0.60, lw=0.4, label=r'Gaussian state ($\sigma_0 = 3$)', zorder=3)
    ax.plot(t_axis, s_loc, color='black', alpha=0.70, lw=0.4, label='Localized', zorder=4)
    
    # Rótulo do painel
    ax.text(0.04, 0.90, "(h)", transform=ax.transAxes, fontsize=13)
    
    ax.set_xlim(0, 500)
    ax.set_ylim(0, 60)
    ax.set_xticks(np.arange(0, 501, 100))
    ax.set_yticks(np.arange(0, 61, 10))
    
    ax.set_xlabel(r"Time Steps $t \, (\times 10^3)$", fontsize=12)
    ax.set_ylabel(r"$\sigma$", fontsize=13)
    ax.legend(frameon=False, loc="upper right", fontsize=9)
    
    plt.tight_layout()
    plt.savefig("fig2_panel_h.png", dpi=300)
    print("Salvo: fig2_panel_h.png")
    plt.close()

if __name__ == "__main__":
    main()

