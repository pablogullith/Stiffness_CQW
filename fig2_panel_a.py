r"""
Reprodução do Painel (a) da FIG. 2
'Stiffness in Confined Quantum Walks'
Distribuição de probabilidade em t \approx T/4.
"""

import numpy as np
import matplotlib.pyplot as plt
from qw_simulation import load_or_run_simulation, setup_matplotlib_style

def main():
    setup_matplotlib_style()
    data = load_or_run_simulation()
    
    j = data['j_full']
    p_loc = data['snaps_loc'][0]
    p_gauss = data['snaps_gauss'][0]
    p_unif = data['snaps_unif'][0]
    
    fig, ax = plt.subplots(figsize=(6, 4.2), dpi=200)
    
    # Linha da distribuição uniforme teórica de referência
    ax.axhline(1.0 / 101, color='gray', linestyle='--', lw=0.8, zorder=1)
    
    # 1. Truncated Uniform state (Azul com preenchimento)
    ax.fill_between(j, 0, p_unif, color='#4169e1', alpha=0.35, zorder=2)
    ax.plot(j, p_unif, color='blue', lw=1.2, label='Truncated Uniform state', zorder=3)
    
    # 2. Gaussian state (sigma_0 = 3) (Vermelho com preenchimento)
    ax.fill_between(j, 0, p_gauss, color='red', alpha=0.30, zorder=4)
    ax.plot(j, p_gauss, color='red', lw=1.2, label=r'Gaussian state ($\sigma_0 = 3$)', zorder=5)
    
    # 3. Localized state (Preto, linha contínua)
    ax.plot(j, p_loc, color='black', lw=1.0, label='Localized', zorder=6)
    
    # Setas indicando propagação dominante para fora
    seta_kw = dict(arrowprops=dict(arrowstyle="-|>", lw=1.3, color="black"))
    ax.annotate("", xy=(-37, 0.068), xytext=(-27, 0.068), **seta_kw)
    ax.annotate("", xy=(37, 0.068), xytext=(27, 0.068), **seta_kw)
    
    # Rótulos do painel e do tempo
    ax.text(0.04, 0.90, "(a)", transform=ax.transAxes, fontsize=13)
    ax.text(0.50, 0.90, r"$t \approx T/4$", transform=ax.transAxes, ha='center', fontsize=13)
    
    ax.set_xlim(-50, 50)
    ax.set_ylim(0.00, 0.14)
    ax.set_xticks([-40, -20, 0, 20, 40])
    ax.set_yticks(np.arange(0.00, 0.15, 0.02))
    ax.yaxis.set_major_formatter(plt.FormatStrFormatter('%.2f'))
    
    ax.set_xlabel(r"Position $j$", fontsize=12)
    ax.set_ylabel(r"Probability", fontsize=12)
    ax.legend(frameon=False, loc="upper right", fontsize=9)
    
    plt.tight_layout()
    plt.savefig("fig2_panel_a.png", dpi=300)
    print("Salvo: fig2_panel_a.png")
    plt.close()

if __name__ == "__main__":
    main()

