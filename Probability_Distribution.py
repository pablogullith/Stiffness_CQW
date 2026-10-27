# Stiffness in Confined Quantum Walks
# Pablo G. M. Dantas, Rafael Vieira e Edgard P. M. Amorim
"""
Caminhada quantica de Hadamard confinada em N=101 sitios.

    - Hadamard nos sitios internos;
    - NOT (Pauli-X) como coin nas duas bordas;
    - deslocamento condicional apos a coin;
    - amplitudes que sairiam da rede sao descartadas.

Estado inicial:

    |psi(0)> = (|L> + i|R>) / sqrt(2)

localizado em j=0.

Tempos:
    41
    82
    122
    163
    1630
    16300
    500000
"""

# Bibliotecas
import os
import numpy as np
import matplotlib.pyplot as plt
from caminhada import estado_inicial, passo

# PARAMETROS
a = 51
N = 2 * a + 1
j = np.arange(-a, a + 1)
origin = a
SNAP_TIMES = [41, 82, 122, 163, 1630, 16300, 500000]
# Arquivos
PREFIX = "DP_CQW_N=101_"
# MATRIZES
# Coin de Hadamard para os sitios internos
H = np.array([[1, 1], [1, -1]], dtype=complex) / np.sqrt(2.0)
# NOT / Pauli-X para as duas bordas
X = np.array([[0, 1], [1, 0]], dtype=complex)


# CARREGA OS DAT
def try_load_dat_files():

    data = {}
    posicoes = None

    for t in SNAP_TIMES:

        fname = f"{PREFIX}{t}.dat"

        if not os.path.exists(fname):
            return None, None

        try:

            arr = np.loadtxt(fname)

        except Exception as e:

            print(f"Erro ao ler {fname}: {e}")
            return None, None

        if arr.ndim != 2 or arr.shape[1] < 2:

            print(f"Formato invalido em {fname}")
            return None, None

        # primeira coluna = j
        # segunda coluna = 100 * P(j,t)
        pos = arr[:, 0].astype(int)
        prob = arr[:, 1] / 100.0

        if posicoes is None:
            posicoes = pos

        data[t] = prob

    return posicoes, data


# SIMULACAO
def simulate_confined_hadamard_walk():

    psiL, psiR = estado_inicial(N)

    data = {}

    t_max = max(SNAP_TIMES)

    for t in range(1, t_max + 1):

        psiL, psiR = passo(psiL, psiR)

        if t in SNAP_TIMES:

            prob = np.abs(psiL) ** 2 + np.abs(psiR) ** 2

            norma = prob.sum()

            print(f"t={t:5d} | " f"norma={norma:.15f}")

            data[t] = prob.copy()

    return data


# CARREGA OU SIMULA
print("=" * 72)
print("CAMINHADA QUANTICA CONFINADA")
print("=" * 72)
print()

posicoes_reais, data = try_load_dat_files()


if data is not None:

    fonte = "arquivos .dat existentes"

    j = posicoes_reais

    print("Os seis arquivos .dat foram encontrados.")
    print("Dados carregados diretamente.")
    print()

else:

    fonte = "simulacao reconstruida"

    print("Os .dat nao foram encontrados.")
    print("Executando a dinamica reconstruida...")
    print()

    data = simulate_confined_hadamard_walk()

# SALVA OS DAT
print()
print("=" * 72)
print("SALVANDO OS ARQUIVOS .DAT")
print("=" * 72)
print()

for t in SNAP_TIMES:

    fname = f"{PREFIX}{t}.dat"
    arr = np.column_stack((j, data[t] * 100.0))
    np.savetxt(fname, arr, fmt=["%d", "%.16E"])
    print(f"OK -> {fname}")

# COMPARACAO / VERIFICACAO NUMERICA
print()
print("=" * 72)
print("VERIFICACAO DOS ARQUIVOS")
print("=" * 72)
print()

for t in SNAP_TIMES:

    fname = f"{PREFIX}{t}.dat"

    arr = np.loadtxt(fname)

    p_arquivo = arr[:, 1] / 100.0

    p_sim = data[t]

    erro = p_arquivo - p_sim

    print(
        f"t={t:5d} | "
        f"soma={p_arquivo.sum():.15f} | "
        f"erro_max={np.max(np.abs(erro)):.3e}"
    )


# DISTRIBUICAO UNIFORME
p_unif = (1.0 / (a + 1) + 1.0 / a) / 2.0
probability_color = "#007C91"
uniform_color = "#E69F00"
arrow_color = "#34495E"

# CONFIGURACAO DA FIGURA

plt.rcParams.update(
    {
        "font.family": "serif",
        "mathtext.fontset": "stix",
        "font.size": 12,
        "axes.linewidth": 1.0,
    }
)

# PAINEIS
fig, axes = plt.subplots(2, 3, figsize=(9.5, 6.2))


paineis = [
    ("a", 41, r"$t \approx T/4$", 0.14, False, "fora"),
    ("b", 82, r"$t \approx T/2$", 0.14, False, None),
    ("c", 122, r"$t \approx 3T/4$", 0.14, False, "dentro"),
    ("d", 163, r"$t \approx T$", 0.08, True, None),
    ("e", 1630, r"$t \approx 10T$", 0.08, True, None),
    ("f", 16300, r"$t \approx 100T$", 0.08, True, None),
]


# DESENHA OS PAINEIS
for ax, (rotulo, t, ttexto, ymax, mostra_unif, setas) in zip(axes.flat, paineis):

    # Distribuicao
    ax.plot(j, data[t], color=probability_color, lw=0.9)
    # Linha uniforme
    if mostra_unif:

        ax.axhline(p_unif, color=uniform_color, linestyle="--", lw=1.0)

    # Limites
    ax.set_xlim(-50, 50)
    ax.set_ylim(0, ymax)
    # Ticks
    ax.tick_params(direction="in", top=True, right=True)
    # Letra
    ax.text(
        0.92,
        0.90,
        f"({rotulo})",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=13,
    )

    # Tempo
    ax.text(
        0.50, 0.90, ttexto, transform=ax.transAxes, ha="center", va="top", fontsize=12
    )

    # Setas
    seta_kw = dict(arrowprops=dict(arrowstyle="-|>", lw=1.3, color=arrow_color))

    if setas == "fora":

        ax.annotate("", xy=(-46, ymax * 0.5), xytext=(-33, ymax * 0.5), **seta_kw)

        ax.annotate("", xy=(46, ymax * 0.5), xytext=(33, ymax * 0.5), **seta_kw)

    elif setas == "dentro":

        ax.annotate("", xy=(-4, ymax * 0.5), xytext=(-17, ymax * 0.5), **seta_kw)

        ax.annotate("", xy=(4, ymax * 0.5), xytext=(17, ymax * 0.5), **seta_kw)

# ROTULOS GERAIS
fig.text(0.012, 0.5, "Probability", va="center", rotation="vertical", fontsize=14)
fig.text(0.5, 0.015, "Position $j$", ha="center", fontsize=14)

# AJUSTES
fig.subplots_adjust(
    left=0.075, right=0.985, top=0.96, bottom=0.10, wspace=0.28, hspace=0.20
)

# SALVA FIGURA
png_file = "Fig2_a-f_colorida.png"
pdf_file = "Fig2_a-f_colorida.pdf"
fig.savefig(png_file, dpi=300)
fig.savefig(pdf_file)
# FINAL
print()
print("=" * 72)
print("CONCLUIDO")
print("=" * 72)
print()
print(f"Fonte dos dados: {fonte}")
print()
print("Arquivos .dat:")
for t in SNAP_TIMES:
    print(f"  DP_CQW_N=101_{t}.dat")

print()
print("Figuras:")
print(f"  {png_file}")
print(f"  {pdf_file}")
print()
print("=" * 72)
plt.show()
