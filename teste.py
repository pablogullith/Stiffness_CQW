# Stiffness in Confined Quantum Walks
# Pablo G. M. Dantas, Rafael Vieira e Edgard P. M. Amorim
"""
Caminhada quântica de Hadamard confinada em N=101 sítios.
Uso da formulação analítica (espectral) baseada em modos normais.

    - Hadamard nos sítios internos;
    - NOT (Pauli-X) como coin nas duas bordas;
    - deslocamento condicional após a coin;
    - amplitudes que sairiam da rede são descartadas.

Estado inicial:
    |psi(0)> = (|L> + i|R>) / sqrt(2)
localizado em j=0 (mapeado para o sítio central da rede).

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

# PARAMETROS
a = 51
N = 2 * a + 1
dim = 2 * N
j = np.arange(-a, a + 1)
origin = a
SNAP_TIMES = [41, 82, 122, 163, 1630, 16300, 500000]
# Arquivos
PREFIX = "DP_CQW_N=101_"

# MATRIZES DO SISTEMA (Base Analítica / Operador Unitário U)
C = np.zeros((dim, dim), dtype=complex)
C[0:2, 0:2] = np.array([[0, 1], [1, 0]], dtype=complex)  # Borda esquerda (NOT)
C[dim - 2 : dim, dim - 2 : dim] = np.array(
    [[0, 1], [1, 0]], dtype=complex
)  # Borda direita (NOT)

H = np.array([[1, 1], [1, -1]], dtype=complex) / np.sqrt(2.0)  # Hadamard interna
for i in range(1, N - 1):
    C[2 * i : 2 * i + 2, 2 * i : 2 * i + 2] = H

S = np.zeros((dim, dim), dtype=complex)
S[1, 0] = 1
S[3, 1] = 1
S[dim - 4, dim - 2] = 1
S[dim - 2, dim - 1] = 1
for i in range(1, N - 1):
    S[2 * (i - 1), 2 * i] = 1
    S[2 * (i + 1) + 1, 2 * i + 1] = 1

# Operador de Evolução Global U = S @ C
U = S @ C

# DECOMPOSIÇÃO ESPECTRAL ANALÍTICA (Modos Normais e Frequências)
print("Calculando autovalores e autovetores do operador de evolução U...")
eigenvalues, eigenvectors = np.linalg.eig(U)
omega = np.angle(eigenvalues)  # Frequências omega_j
Phi = eigenvectors  # Modos normais j (colunas)

# Separando as componentes espaciais dos modos para L e R
Phi_L = Phi[0::2, :]  # shape (N, dim)
Phi_R = Phi[1::2, :]  # shape (N, dim)

# PRÉ-COMPUTAÇÃO DA MATRIZ DE DENSIDADE ESPACIAL MODAL
# Evita loop temporal pesado: densidade_jmx[j, m, x]
print("Pré-computando acoplamentos espaciais modais...")
density_matrix_spatial = np.einsum("xj,xm->jmx", Phi_L, np.conj(Phi_L)) + np.einsum(
    "xj,xm->jmx", Phi_R, np.conj(Phi_R)
)


# ESTADO INICIAL
def get_initial_state():
    psi_0 = np.zeros(dim, dtype=complex)
    # j = 0 corresponde ao índice central no vetor de posições de -a até a
    x_inicial_idx = origin  # índice central
    psi_0[2 * x_inicial_idx] = 1.0 / np.sqrt(2.0)
    psi_0[2 * x_inicial_idx + 1] = 1j / np.sqrt(2.0)
    return psi_0


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

        pos = arr[:, 0].astype(int)
        prob = arr[:, 1] / 100.0

        if posicoes is None:
            posicoes = pos

        data[t] = prob

    return posicoes, data


# SIMULAÇÃO USANDO A FORMULAÇÃO ANALÍTICA MODAL
def simulate_confined_hadamard_walk_analytic():
    psi_0 = get_initial_state()

    # Projeção do estado inicial para encontrar os coeficientes c_j
    psi_L_0 = psi_0[0::2]
    psi_R_0 = psi_0[1::2]

    c = np.zeros(dim, dtype=complex)
    for j in range(dim):
        c[j] = np.sum(np.conj(Phi_L[:, j]) * psi_L_0 + np.conj(Phi_R[:, j]) * psi_R_0)

    data = {}
    t_max = max(SNAP_TIMES)

    print(f"Executando simulação analítica até t_max = {t_max}...")

    for t in range(1, t_max + 1):
        # Fatores de fase temporal analíticos: exp(i * (omega_j - omega_m) * t)
        phase_matrix = np.exp(1j * outer_diff(omega, omega) * t)

        # Matriz de coeficientes acoplados c_j c_m^* exp(...)
        C_matrix = np.outer(c, np.conj(c)) * phase_matrix

        # Probabilidade exata via expansão modal: P(x,t) = \sum_{j,m} C_jm * [...]
        prob = np.real(np.einsum("jm,jmx->x", C_matrix, density_matrix_spatial))

        norma = prob.sum()

        if t in SNAP_TIMES:
            print(f"t={t:5d} | norma={norma:.15f}")
            data[t] = prob.copy()

    return data


def outer_diff(a, b):
    return a[:, np.newaxis] - b[np.newaxis, :]


# CARREGA OU SIMULA
print("=" * 72)
print("CAMINHADA QUÂNTICA CONFINADA (FORMULAÇÃO ANALÍTICA)")
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
    fonte = "simulação analítica reconstruída"
    print("Os .dat não foram encontrados.")
    print("Executando a dinâmica via expansão modal analítica...")
    print()
    data = simulate_confined_hadamard_walk_analytic()

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
    # Distribuição
    ax.plot(j, data[t], color="black", lw=0.8)

    # Linha uniforme
    if mostra_unif:
        ax.axhline(p_unif, color="red", linestyle="--", lw=1.0)

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
    seta_kw = dict(arrowprops=dict(arrowstyle="-|>", lw=1.3, color="black"))

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
png_file = "Fig2_a-f.png"
pdf_file = "Fig2_a-f.pdf"
fig.savefig(png_file, dpi=300)
fig.savefig(pdf_file)

# FINAL
print()
print("=" * 72)
print("CONCLUÍDO")
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
