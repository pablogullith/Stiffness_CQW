r"""
Reprodução do Painel (l) da FIG. 2
'Stiffness in Confined Quantum Walks'
Funções densidade de probabilidade empíricas da coerência C(t) no intervalo 2x10^5 < t <= 5x10^5
com os respectivos ajustes Beta escalados: p(C) = 4 * Beta(4C; alpha, beta).
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import beta
from qw_simulation import load_or_run_simulation, setup_matplotlib_style

def main():
    setup_matplotlib_style()
    data = load_or_run_simulation()
    
    # Janela de análise de tempos longos: 2*10^5 < t <= 5*10^5
    idx = np.arange(200_001, 500_001)
    
    c_loc = data['c_loc'][idx]
    c_gauss = data['c_gauss'][idx]
    c_unif = data['c_unif'][idx]
    
    # Ajustes Beta para x = 4*C in [0, 1]
    # Parâmetros analíticos/obtidos no artigo:
    # Localizado:       alpha ~ 4.046, beta ~ 18.296
    # Gaussiano:        alpha ~ 0.501, beta ~ 5.068
    # Uniforme trunc.:  alpha ~ 0.607, beta ~ 1.444
    a_loc, b_loc, _, _ = beta.fit(4.0 * c_loc, floc=0, fscale=1)
    a_gauss, b_gauss, _, _ = beta.fit(4.0 * c_gauss, floc=0, fscale=1)
    a_unif, b_unif, _, _ = beta.fit(4.0 * c_unif, floc=0, fscale=1)
    
    fig, ax = plt.subplots(figsize=(6, 4.2), dpi=200)
    
    c_grid = np.linspace(1e-5, 0.2499, 1000)
    bins = np.linspace(0.0, 0.25, 120)
    
    # 1. Truncated Uniform state (Histograma azul + ajuste azul)
    ax.hist(c_unif, bins=bins, density=True, color='#5c82ff', alpha=0.45, label='Uniform (empírico)', zorder=2)
    pdf_unif = 4.0 * beta.pdf(4.0 * c_grid, a_unif, b_unif)
    ax.plot(c_grid, pdf_unif, color='blue', lw=2.0, label=f'Scaled Beta Fit (U: $\\alpha$={a_unif:.2f}, $\\beta$={b_unif:.2f})', zorder=5)
    
    # 2. Gaussian state (Histograma vermelho + ajuste vermelho)
    ax.hist(c_gauss, bins=bins, density=True, color='#ff6b6b', alpha=0.45, label='Gaussian (empírico)', zorder=3)
    pdf_gauss = 4.0 * beta.pdf(4.0 * c_grid, a_gauss, b_gauss)
    ax.plot(c_grid, pdf_gauss, color='red', lw=2.0, label=f'Scaled Beta Fit (G: $\\alpha$={a_gauss:.2f}, $\\beta$={b_gauss:.2f})', zorder=6)
    
    # 3. Localized state (Histograma cinza preenchido + ajuste preto)
    ax.hist(c_loc, bins=bins, density=True, color='#7f7f7f', alpha=0.60, label='Localized (empírico)', zorder=4)
    pdf_loc = 4.0 * beta.pdf(4.0 * c_grid, a_loc, b_loc)
    ax.plot(c_grid, pdf_loc, color='black', lw=2.0, label=f'Scaled Beta Fit (L: $\\alpha$={a_loc:.2f}, $\\beta$={b_loc:.2f})', zorder=7)
    
    # Rótulos do painel e título interno
    ax.text(0.04, 0.90, "(l)", transform=ax.transAxes, fontsize=13)
    ax.text(0.50, 0.90, "Scaled Beta Fit", transform=ax.transAxes, ha='center', fontsize=13)
    
    ax.set_xlim(0.00, 0.25)
    ax.set_ylim(0, 100)
    ax.set_xticks(np.arange(0.00, 0.26, 0.05))
    ax.set_yticks(np.arange(0, 101, 20))
    ax.xaxis.set_major_formatter(plt.FormatStrFormatter('%.2f'))
    
    ax.set_xlabel(r"$C(t > 2\times 10^5)$", fontsize=12)
    ax.set_ylabel(r"PDF", fontsize=12)
    ax.legend(frameon=False, loc="upper right", fontsize=8)
    
    plt.tight_layout()
    plt.savefig("fig2_panel_l.png", dpi=300)
    print("Salvo: fig2_panel_l.png")
    plt.close()

if __name__ == "__main__":
    main()

