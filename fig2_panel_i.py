r"""
Reprodução do Painel (i) da FIG. 2
'Stiffness in Confined Quantum Walks'
Funções densidade de probabilidade empíricas de \sigma(t) no intervalo 2x10^5 < t <= 5x10^5
com os respectivos ajustes Gaussianos.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from qw_simulation import load_or_run_simulation, setup_matplotlib_style

def main():
    setup_matplotlib_style()
    data = load_or_run_simulation()
    
    # Janela de análise de tempos longos: 2*10^5 < t <= 5*10^5
    idx = np.arange(200_001, 500_001)
    
    s_loc = data['s_loc'][idx]
    s_gauss = data['s_gauss'][idx]
    s_unif = data['s_unif'][idx]
    
    # Ajustes Gaussianos
    mu_loc, std_loc = norm.fit(s_loc)
    mu_gauss, std_gauss = norm.fit(s_gauss)
    mu_unif, std_unif = norm.fit(s_unif)
    
    fig, ax = plt.subplots(figsize=(6, 4.2), dpi=200)
    
    x_axis = np.linspace(10, 50, 500)
    bins = np.linspace(15, 45, 100)
    
    # 1. Truncated Uniform state (Histograma azul + ajuste azul)
    ax.hist(s_unif, bins=bins, density=True, color='#5c82ff', alpha=0.45, label='Uniform (empírico)', zorder=2)
    
    # 2. Gaussian state (Histograma vermelho/salmão)
    ax.hist(s_gauss, bins=bins, density=True, color='#ff6b6b', alpha=0.45, label='Gaussian (empírico)', zorder=3)
    
    # Curva de ajuste para Uniforme/Gaussiana (praticamente indistinguíveis)
    ax.plot(x_axis, norm.pdf(x_axis, mu_unif, std_unif), color='blue', lw=2.0, label='Gaussian Fit (Uniform/Gauss)', zorder=5)
    
    # 3. Localized state (Histograma cinza preenchido + ajuste preto)
    ax.hist(s_loc, bins=bins, density=True, color='#7f7f7f', alpha=0.60, label='Localized (empírico)', zorder=4)
    ax.plot(x_axis, norm.pdf(x_axis, mu_loc, std_loc), color='black', lw=2.0, label='Gaussian Fit (Localized)', zorder=6)
    
    # Rótulos do painel e título interno
    ax.text(0.04, 0.90, "(i)", transform=ax.transAxes, fontsize=13)
    ax.text(0.50, 0.90, "Gaussian Fit", transform=ax.transAxes, ha='center', fontsize=13)
    
    ax.set_xlim(10, 50)
    ax.set_ylim(0.00, 0.25)
    ax.set_xticks(np.arange(10, 51, 10))
    ax.set_yticks(np.arange(0.00, 0.26, 0.05))
    ax.yaxis.set_major_formatter(plt.FormatStrFormatter('%.2f'))
    
    ax.set_xlabel(r"$\sigma(t > 2\times 10^5)$", fontsize=12)
    ax.set_ylabel(r"PDF", fontsize=12)
    ax.legend(frameon=False, loc="upper right", fontsize=8)
    
    plt.tight_layout()
    plt.savefig("fig2_panel_i.png", dpi=300)
    print("Salvo: fig2_panel_i.png")
    plt.close()

if __name__ == "__main__":
    main()

