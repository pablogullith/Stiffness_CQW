""" "
#Stiffness in Confined Quantum Walks
# Pablo G. M. Dantas, Rafael Vieira e Edgard P. M. Amorim

# Caminhada quântica de Hadamard confinada
# N = 101, 201 e 301 sítios

# Estado inicial:
#     |psi(0)> = (|L> + i|R>) / sqrt(2)
# localizado em j = 0.

# Coin:
#     - Hadamard nos sítios internos
#     - Pauli-X nas duas bordas

# Shift:
#     - |L> -> j-1
#     - |R> -> j+1

# Amplitudes que sairiam da rede são descartadas.

# O gráfico mostra:

#     sigma(t) = sqrt(<j^2> - <j>^2)

# com a distribuição normalizada pela norma sobrevivente.
"""

# BIBLIOTECAS
import numpy as np
import matplotlib.pyplot as plt
from caminhada import estado_inicial, passo

# PARÂMETROS
N_VALUES = [101, 201, 301]
T_MAX = 1000


# SIMULAÇÃO E CÁLCULO DA DISPERSÃO
def simulate_dispersion(N, t_max):
    """
    Simula a caminhada quântica para uma rede de tamanho N.

    Retorna:

        times
        norm
        mean_j
        mean_j2
        variance
        sigma

    A distribuição é normalizada pela norma sobrevivente
    em cada instante.
    """
    a = (N - 1) // 2
    # Posições físicas:
    # j = -a, ..., 0, ..., +a
    j = np.arange(-a, a + 1, dtype=float)
    # Estado inicial
    psiL, psiR = estado_inicial(N)

    # Arrays para armazenar os resultados
    times = np.arange(t_max + 1)
    norm = np.zeros(t_max + 1)
    mean_j = np.zeros(t_max + 1)
    mean_j2 = np.zeros(t_max + 1)
    variance = np.zeros(t_max + 1)
    sigma = np.zeros(t_max + 1)
    # t = 0
    prob = np.abs(psiL) ** 2 + np.abs(psiR) ** 2
    Z = prob.sum()
    norm[0] = Z
    # Distribuição normalizada
    p_norm = prob / Z

    # Primeiro momento
    mean_j[0] = np.sum(j * p_norm)
    # Segundo momento
    mean_j2[0] = np.sum(j**2 * p_norm)

    # Variância
    variance[0] = mean_j2[0] - mean_j[0] ** 2
    variance[0] = max(variance[0], 0.0)
    # Desvio padrão
    sigma[0] = np.sqrt(variance[0])
    # EVOLUÇÃO TEMPORAL

    for t in range(1, t_max + 1):

        # Um passo da caminhada
        psiL, psiR = passo(psiL, psiR)
        # Distribuição de probabilidade
        prob = np.abs(psiL) ** 2 + np.abs(psiR) ** 2
        Z = prob.sum()
        norm[t] = Z

        # Normalização

        if Z > 0.0:

            p_norm = prob / Z
            # Primeiro momento
            mean_j[t] = np.sum(j * p_norm)
            # Segundo momento
            mean_j2[t] = np.sum(j**2 * p_norm)
            # Variância
            variance[t] = mean_j2[t] - mean_j[t] ** 2
            variance[t] = max(variance[t], 0.0)

            # Desvio padrão
            sigma[t] = np.sqrt(variance[t])

    return (times, norm, mean_j, mean_j2, variance, sigma)


# SIMULAÇÃO PARA N = 101, 201 E 301

results = {}
print("=" * 72)
print("CAMINHADA QUÂNTICA CONFINADA")
print("DISPERSÃO PARA N = 101, 201 E 301")
print("=" * 72)
print()

for N in N_VALUES:

    print(f"Simulando N = {N} ...")

    results[N] = simulate_dispersion(N, T_MAX)

    times, norm, mean_j, mean_j2, variance, sigma = results[N]

    print(f"  Norma final     = " f"{norm[-1]:.15f}")

    print(f"  <j> final       = " f"{mean_j[-1]:.15f}")

    print(f"  Variância final = " f"{variance[-1]:.15f}")

    print(f"  Sigma final     = " f"{sigma[-1]:.15f}")

    print()


# CONFIGURAÇÃO GRÁFICA

plt.rcParams.update(
    {
        "font.family": "serif",
        "mathtext.fontset": "stix",
        "font.size": 12,
        "axes.linewidth": 1.0,
    }
)


# GRÁFICO DA DISPERSÃO
fig, ax = plt.subplots(figsize=(8.5, 5.5))
# Curvas para os três tamanhos
for N in N_VALUES:

    times = results[N][0]
    sigma = results[N][5]

    ax.plot(times, sigma, lw=1.5, label=rf"$N={N}$")

# RÓTULOS
ax.set_xlabel(r"Time $t$", fontsize=14)
ax.set_ylabel(r"Standard deviation $\sigma(t)$", fontsize=14)
# LIMITES
ax.set_xlim(0, T_MAX)
# TICKS
ax.tick_params(direction="in", top=True, right=True)
# LEGENDA
ax.legend(frameon=False)
# AJUSTES
fig.subplots_adjust(left=0.11, right=0.97, bottom=0.12, top=0.97)
# SALVA FIGURA
png_file = "Dispersion_sigma_" "N101_N201_N301.png"
pdf_file = "Dispersion_sigma_" "N101_N201_N301.pdf"
fig.savefig(png_file, dpi=300)
fig.savefig(pdf_file)
# FINAL
print("=" * 72)
print("CONCLUÍDO")
print("=" * 72)
print()
print("Figura PNG:")
print(f"  {png_file}")
print()
print("Figura PDF:")
print(f"  {pdf_file}")
print()
plt.show()
