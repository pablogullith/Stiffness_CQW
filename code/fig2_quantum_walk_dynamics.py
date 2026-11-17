r"""
FIG. 2
'Stiffness in Confined Quantum Walks'
Pablo G. M. Dantas, Rafael Vieira, Edgard P. M. Amorim

Generates the 12-panel figure (a)-(l).
"""

import numpy as np
import os
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy.stats import norm, beta
from qw_simulation import load_or_run_simulation, setup_matplotlib_style

IMAGE_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "images")

def main():
    setup_matplotlib_style()
    data = load_or_run_simulation()
    
    j = data['j_full']
    snap_loc = data['snaps_loc']
    snap_gauss = data['snaps_gauss']
    snap_unif = data['snaps_unif']
    
    s_loc = data['s_loc']
    s_gauss = data['s_gauss']
    s_unif = data['s_unif']
    
    c_loc = data['c_loc']
    c_gauss = data['c_gauss']
    c_unif = data['c_unif']
    
    fig, axes = plt.subplots(4, 3, figsize=(11.5, 9.5), dpi=300)
    
    # Standardized colors identical to the paper
    col_loc = "black"
    col_gauss = "red"
    col_gauss_fill = "#ff6b6b"
    col_unif = "blue"
    col_unif_fill = "#4169e1"
    col_gray_fill = "#7f7f7f"
    
    # ROWS 1 AND 2: Panels (a) to (f) - Probability Distributions
    
    snap_configs = [
        # (row, col, idx_snap, label, time_str, arrows)
        (0, 0, 0, "(a)", r"$t \approx T/4$", "outward"),
        (0, 1, 1, "(b)", r"$t \approx T/2$", None),
        (0, 2, 2, "(c)", r"$t \approx 3T/4$", "inward"),
        (1, 0, 3, "(d)", r"$t \approx T$", None),
        (1, 1, 4, "(e)", r"$t \approx 10T$", None),
        (1, 2, 5, "(f)", r"$t \approx 100T$", None),
    ]
    
    arrow_kw = dict(arrowprops=dict(arrowstyle="-|>", lw=1.2, color="black"))
    
    for r, c, s_idx, lbl, t_str, arrows in snap_configs:
        ax = axes[r, c]
        p_u = snap_unif[s_idx]
        p_g = snap_gauss[s_idx]
        p_l = snap_loc[s_idx]
        
        # Uniform reference
        ax.axhline(1.0 / 101, color='gray', linestyle='--', lw=0.8, zorder=1)
        
        # Uniform
        ax.fill_between(j, 0, p_u, color=col_unif_fill, alpha=0.35, zorder=2)
        ax.plot(j, p_u, color=col_unif, lw=1.1, zorder=3)
        
        # Gaussian
        ax.fill_between(j, 0, p_g, color=col_gauss, alpha=0.30, zorder=4)
        ax.plot(j, p_g, color=col_gauss, lw=1.1, zorder=5)
        
        # Localized
        ax.plot(j, p_l, color=col_loc, lw=0.9, zorder=6)
        
        # Arrows
        if arrows == "outward":
            ax.annotate("", xy=(-37, 0.068), xytext=(-27, 0.068), **arrow_kw)
            ax.annotate("", xy=(37, 0.068), xytext=(27, 0.068), **arrow_kw)
        elif arrows == "inward":
            ax.annotate("", xy=(-10, 0.068), xytext=(-20, 0.068), **arrow_kw)
            ax.annotate("", xy=(10, 0.068), xytext=(20, 0.068), **arrow_kw)
            
        ax.text(0.05, 0.88, lbl, transform=ax.transAxes, fontsize=12)
        ax.text(0.50, 0.88, t_str, transform=ax.transAxes, ha='center', fontsize=12)
        
        ax.set_xlim(-50, 50)
        ax.set_ylim(0.00, 0.14)
        ax.set_xticks([-40, -20, 0, 20, 40])
        ax.set_yticks(np.arange(0.00, 0.15, 0.02))
        ax.yaxis.set_major_formatter(plt.FormatStrFormatter('%.2f'))
        
        if c != 0:
            ax.set_yticklabels([])
        if r == 0:
            ax.set_xticklabels([])
            
    # General labels for rows 1 and 2
    axes[1, 1].set_xlabel(r"Position $j$", fontsize=13)
    fig.text(0.020, 0.72, r"Probability", va='center', rotation='vertical', fontsize=14)
    
    
    # ROW 3: Panels (g), (h), (i) - Standard Deviation sigma(t)
    # (g) Early-time sigma
    ax_g = axes[2, 0]
    t_early = np.arange(1001) / 1000.0
    ax_g.plot(t_early, s_unif[:1001], color=col_unif, lw=1.3, zorder=2)
    ax_g.plot(t_early, s_gauss[:1001], color=col_gauss, lw=1.3, zorder=3)
    ax_g.plot(t_early, s_loc[:1001], color=col_loc, lw=1.1, zorder=4)
    ax_g.text(0.05, 0.88, "(g)", transform=ax_g.transAxes, fontsize=12)
    ax_g.set_xlim(0.0, 1.0)
    ax_g.set_ylim(0, 60)
    ax_g.set_xticks(np.arange(0.0, 1.1, 0.2))
    ax_g.set_yticks(np.arange(0, 61, 10))
    ax_g.set_ylabel(r"$\sigma$", fontsize=13)
    ax_g.set_xticklabels([])
    
    # (h) Long-time sigma
    ax_h = axes[2, 1]
    t_long = np.arange(500001) / 1000.0
    ax_h.plot(t_long, s_unif, color='#5c82ff', alpha=0.55, lw=0.35, zorder=2)
    ax_h.plot(t_long, s_gauss, color='#ff5050', alpha=0.60, lw=0.35, zorder=3)
    ax_h.plot(t_long, s_loc, color=col_loc, alpha=0.70, lw=0.35, zorder=4)
    ax_h.text(0.05, 0.88, "(h)", transform=ax_h.transAxes, fontsize=12)
    ax_h.set_xlim(0, 500)
    ax_h.set_ylim(0, 60)
    ax_h.set_xticks(np.arange(0, 501, 100))
    ax_h.set_yticks(np.arange(0, 61, 10))
    ax_h.set_yticklabels([])
    ax_h.set_xticklabels([])
    
    # (i) PDF of sigma with Gaussian Fit
    ax_i = axes[2, 2]
    idx_tail = np.arange(200_001, 500_001)
    s_loc_tail = s_loc[idx_tail]
    s_gauss_tail = s_gauss[idx_tail]
    s_unif_tail = s_unif[idx_tail]
    
    mu_loc, std_loc = norm.fit(s_loc_tail)
    mu_unif, std_unif = norm.fit(s_unif_tail)
    
    x_grid_s = np.linspace(10, 50, 500)
    bins_s = np.linspace(15, 45, 100)
    
    ax_i.hist(s_unif_tail, bins=bins_s, density=True, color='#5c82ff', alpha=0.45, zorder=2)
    ax_i.hist(s_gauss_tail, bins=bins_s, density=True, color='#ff6b6b', alpha=0.45, zorder=3)
    ax_i.plot(x_grid_s, norm.pdf(x_grid_s, mu_unif, std_unif), color=col_unif, lw=1.8, zorder=5)
    
    ax_i.hist(s_loc_tail, bins=bins_s, density=True, color=col_gray_fill, alpha=0.60, zorder=4)
    ax_i.plot(x_grid_s, norm.pdf(x_grid_s, mu_loc, std_loc), color=col_loc, lw=1.8, zorder=6)
    
    ax_i.text(0.05, 0.88, "(i)", transform=ax_i.transAxes, fontsize=12)
    ax_i.text(0.50, 0.88, "Gaussian Fit", transform=ax_i.transAxes, ha='center', fontsize=12)
    ax_i.set_xlim(10, 50)
    ax_i.set_ylim(0.00, 0.25)
    ax_i.set_xticks(np.arange(10, 51, 10))
    ax_i.set_yticks(np.arange(0.00, 0.26, 0.05))
    ax_i.yaxis.set_major_formatter(plt.FormatStrFormatter('%.2f'))
    ax_i.yaxis.tick_right()
    ax_i.yaxis.set_label_position("right")
    ax_i.set_ylabel(r"PDF", fontsize=12)
    ax_i.set_xlabel(r"$\sigma(t > 2\times 10^5)$", fontsize=12)
    
    
    # ROW 4: Panels (j), (k), (l) - Coin Coherence C(t)
    # (j) Early-time C
    ax_j = axes[3, 0]
    ax_j.plot(t_early, c_unif[:1001], color=col_unif, lw=1.3, zorder=2)
    ax_j.plot(t_early, c_gauss[:1001], color=col_gauss, lw=1.3, zorder=3)
    ax_j.plot(t_early, c_loc[:1001], color=col_loc, lw=0.9, zorder=4)
    ax_j.text(0.05, 0.88, "(j)", transform=ax_j.transAxes, fontsize=12)
    ax_j.set_xlim(0.0, 1.0)
    ax_j.set_ylim(0.00, 0.30)
    ax_j.set_xticks(np.arange(0.0, 1.1, 0.2))
    ax_j.set_yticks(np.arange(0.00, 0.31, 0.05))
    ax_j.yaxis.set_major_formatter(plt.FormatStrFormatter('%.2f'))
    ax_j.set_ylabel(r"$C$", fontsize=13)
    
    # (k) Long-time C
    ax_k = axes[3, 1]
    ax_k.plot(t_long, c_unif, color='#5c82ff', alpha=0.55, lw=0.35, zorder=2)
    ax_k.plot(t_long, c_gauss, color='#ff5050', alpha=0.60, lw=0.35, zorder=3)
    ax_k.plot(t_long, c_loc, color=col_loc, alpha=0.70, lw=0.35, zorder=4)
    ax_k.text(0.05, 0.88, "(k)", transform=ax_k.transAxes, fontsize=12)
    ax_k.set_xlim(0, 500)
    ax_k.set_ylim(0.00, 0.30)
    ax_k.set_xticks(np.arange(0, 501, 100))
    ax_k.set_yticks(np.arange(0.00, 0.31, 0.05))
    ax_k.set_yticklabels([])
    
    fig.text(0.38, 0.02, r"Time Steps $t \, (\times 10^3)$", ha='center', fontsize=13)
    
    # (l) PDF of C with Scaled Beta Fit
    ax_l = axes[3, 2]
    c_loc_tail = c_loc[idx_tail]
    c_gauss_tail = c_gauss[idx_tail]
    c_unif_tail = c_unif[idx_tail]
    
    a_loc, b_loc, _, _ = beta.fit(4.0 * c_loc_tail, floc=0, fscale=1)
    a_gauss, b_gauss, _, _ = beta.fit(4.0 * c_gauss_tail, floc=0, fscale=1)
    a_unif, b_unif, _, _ = beta.fit(4.0 * c_unif_tail, floc=0, fscale=1)
    
    c_grid = np.linspace(1e-5, 0.2499, 1000)
    bins_c = np.linspace(0.0, 0.25, 120)
    
    ax_l.hist(c_unif_tail, bins=bins_c, density=True, color='#5c82ff', alpha=0.45, zorder=2)
    pdf_unif = 4.0 * beta.pdf(4.0 * c_grid, a_unif, b_unif)
    ax_l.plot(c_grid, pdf_unif, color=col_unif, lw=1.8, zorder=5)
    
    ax_l.hist(c_gauss_tail, bins=bins_c, density=True, color='#ff6b6b', alpha=0.45, zorder=3)
    pdf_gauss = 4.0 * beta.pdf(4.0 * c_grid, a_gauss, b_gauss)
    ax_l.plot(c_grid, pdf_gauss, color=col_gauss, lw=1.8, zorder=6)
    
    ax_l.hist(c_loc_tail, bins=bins_c, density=True, color=col_gray_fill, alpha=0.60, zorder=4)
    pdf_loc = 4.0 * beta.pdf(4.0 * c_grid, a_loc, b_loc)
    ax_l.plot(c_grid, pdf_loc, color=col_loc, lw=1.8, zorder=7)
    
    ax_l.text(0.05, 0.88, "(l)", transform=ax_l.transAxes, fontsize=12)
    ax_l.text(0.50, 0.88, "Scaled Beta Fit", transform=ax_l.transAxes, ha='center', fontsize=12)
    ax_l.set_xlim(0.00, 0.25)
    ax_l.set_ylim(0, 100)
    ax_l.set_xticks(np.arange(0.00, 0.26, 0.05))
    ax_l.set_yticks(np.arange(0, 101, 20))
    ax_l.xaxis.set_major_formatter(plt.FormatStrFormatter('%.2f'))
    ax_l.yaxis.tick_right()
    ax_l.yaxis.set_label_position("right")
    ax_l.set_ylabel(r"PDF", fontsize=12)
    ax_l.set_xlabel(r"$C(t > 2\times 10^5)$", fontsize=12)
    
    
    # Global Top Legend
    legend_elements = [
        Line2D([0], [0], color=col_loc, lw=1.5, label='Localized'),
        Line2D([0], [0], color=col_gauss, lw=1.5, label=r'Gaussian state ($\sigma_0 = 3$)'),
        Line2D([0], [0], color=col_unif, lw=1.5, label='Truncated Uniform state'),
    ]
    fig.legend(
        handles=legend_elements,
        loc='upper center',
        bbox_to_anchor=(0.50, 0.995),
        ncol=3,
        frameon=False,
        fontsize=12
    )
    
    # Fine-tune spacing
    fig.subplots_adjust(
        left=0.085,
        right=0.92,
        top=0.95,
        bottom=0.06,
        wspace=0.18,
        hspace=0.25
    )
    
    #png_out = "Figura2.png"
    pdf_out = os.path.join(IMAGE_DIR, "Figura02.pdf")
    #plt.savefig(png_out, dpi=300)
    plt.savefig(pdf_out)
    #print(f"Complete figure saved to {png_out} and {pdf_out}")
    plt.close()

if __name__ == "__main__":
    main()