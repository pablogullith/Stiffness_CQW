"""
Animação da Caminhada Quântica Confinada
N = 101 sítios, moeda Hadamard (internos) e X (bordas)
Estado inicial: (|L> + i|R>)/sqrt(2) em j=0
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

a = 51
N = 2 * a + 1
j = np.arange(-a, a + 1)


T_MAX = 200          # você pode aumentar para ver mais reflexões
INTERVAL = 50        # ms entre frames

# operadores de moeda
H = np.array([[1, 1], [1, -1]], dtype=complex) / np.sqrt(2.0)   # Hadamard
X = np.array([[0, 1], [1, 0]], dtype=complex)                   # Pauli-X (NOT)

# Funções
def estado_inicial(N, centro):
    """Retorna (psiL, psiR) do estado |L> + i|R> localizado no centro."""
    psiL = np.zeros(N, dtype=complex)
    psiR = np.zeros(N, dtype=complex)
    psiL[centro] = 1.0 / np.sqrt(2.0)
    psiR[centro] = 1j / np.sqrt(2.0)
    return psiL, psiR

def passo(psiL, psiR):
    """Aplica moeda local e deslocamento condicional."""
    # 1) Aplica moeda em cada sítio
    for x in range(N):
        # escolhe operador: X nas bordas, H no interior
        coin = X if (x == 0 or x == N-1) else H
        l_old = psiL[x]
        r_old = psiR[x]
        psiL[x] = coin[0,0] * l_old + coin[0,1] * r_old
        psiR[x] = coin[1,0] * l_old + coin[1,1] * r_old

    # 2) Deslocamento condicional (com descarte nas bordas)
    novoL = np.zeros(N, dtype=complex)
    novoR = np.zeros(N, dtype=complex)

    # Para psiL: move para a esquerda (j-1) se não estiver na borda esquerda
    for x in range(1, N):
        novoL[x-1] += psiL[x]
    # Para psiR: move para a direita (j+1) se não estiver na borda direita
    for x in range(N-1):
        novoR[x+1] += psiR[x]

    return novoL, novoR

# Simulação
centro = a
psiL, psiR = estado_inicial(N, centro)

# Lista para armazenar a distribuição em cada instante
prob_evol = []
for t in range(T_MAX + 1):
    prob = np.abs(psiL)**2 + np.abs(psiR)**2
    prob_evol.append(prob.copy())
    if t < T_MAX:
        psiL, psiR = passo(psiL, psiR)

# Animação
# Configurações de estilo
plt.rcParams.update({
    "font.family": "serif",
    "mathtext.fontset": "stix",
    "font.size": 12,
    "axes.linewidth": 1.0,
})

fig, ax = plt.subplots(figsize=(8, 5))
ax.set_xlim(-50, 50)
ax.set_ylim(0, 0.14)          # ajuste se necessário
ax.set_xlabel("Position $j$")
ax.set_ylabel("Probability")
ax.tick_params(direction="in", top=True, right=True)

# Linha da distribuição uniforme (opcional)
p_unif = 1.0 / N
ax.axhline(p_unif, color='red', linestyle='--', lw=1.0, label='Uniform')

# Linha que será atualizada
line, = ax.plot([], [], color='black', lw=1.2)
time_text = ax.text(0.02, 0.95, '', transform=ax.transAxes, fontsize=12)

def init():
    line.set_data([], [])
    time_text.set_text('')
    return line, time_text

def update(frame):
    line.set_data(j, prob_evol[frame])
    time_text.set_text(f'$t = {frame}$')
    return line, time_text

ani = animation.FuncAnimation(
    fig, update, frames=range(T_MAX + 1),
    init_func=init, blit=True, interval=INTERVAL
)

# Salva como GIF (requer Pillow) ou MP4 (requer ffmpeg)
ani.save('caminhada_quantica_confined.gif', writer='pillow', fps=20)
# ani.save('caminhada_quantica_confined.mp4', writer='ffmpeg', fps=20)

plt.show()