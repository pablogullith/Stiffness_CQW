# Stiffness pelo método analítico (para N = 101, 201, 301)
"""
!!REVISAR STIFFNESS
"""

import numpy as np
import math
import matplotlib.pyplot as plt

def main():
    # Parâmetros comuns
    STIFF_PERIOD = 163
    NUM_STEPS = 1000 * STIFF_PERIOD
    v_max = 1.0 / np.sqrt(2.0)

    N_values = [101, 201, 301]
    plt.figure(figsize=(10, 6))

    for N in N_values:
        print(f"Simulando N = {N} ...")

        # Configuração das paredes e nós
        boundary = (N + 1) // 2
        N_nodes = 2 * boundary + 1
        center_idx = boundary
        dim = 2 * N_nodes
        posicoes = np.arange(-boundary, boundary + 1)

        # Construção dos operadores U = S @ C
        C = np.zeros((dim, dim), dtype=complex)
        S = np.zeros((dim, dim), dtype=complex)

        for i in range(N_nodes):
            if i == 0 or i == N_nodes - 1:
                # Reflexão total nas paredes (moeda especial com q=0)
                C[2*i, 2*i+1] = 1.0
                C[2*i+1, 2*i] = 1.0
            else:
                # Moeda de Hadamard no interior (q=0.5)
                C[2*i, 2*i] = 1/np.sqrt(2)
                C[2*i, 2*i+1] = 1/np.sqrt(2)
                C[2*i+1, 2*i] = 1/np.sqrt(2)
                C[2*i+1, 2*i+1] = -1/np.sqrt(2)

            # Operador de Shift
            if i < N_nodes - 1:
                S[2*(i+1), 2*i] = 1        # a move para a direita
            if i > 0:
                S[2*(i-1)+1, 2*i+1] = 1    # b move para a esquerda

        U = S @ C

        # Estado Inicial
        psi_0 = np.zeros(dim, dtype=complex)
        psi_0[2 * center_idx] = 1.0

        # Operadores de Posição (X e X²)
        X_op = np.diag(np.repeat(posicoes, 2))
        X2_op = np.diag(np.repeat(posicoes**2, 2))

        # Decomposição espectral
        evals, evecs = np.linalg.eig(U)

        c_mu = np.linalg.solve(evecs, psi_0)
        X_munu = np.conj(evecs).T @ X_op @ evecs
        X2_munu = np.conj(evecs).T @ X2_op @ evecs

        n_out = NUM_STEPS // STIFF_PERIOD
        T_arr = np.zeros(n_out, dtype=int)
        Sv_arr = np.zeros(n_out)

        sigma_prev = 0.0
        abs_var = 0.0
        out_idx = 0

        for t in range(1, NUM_STEPS + 1):
            v_t = c_mu * (evals ** t)

            M1_t = np.real(np.vdot(v_t, X_munu @ v_t))
            M2_t = np.real(np.vdot(v_t, X2_munu @ v_t))

            var = M2_t - M1_t**2
            sigma = math.sqrt(max(0.0, var))

            abs_var += abs(sigma - sigma_prev)

            if t % STIFF_PERIOD == 0:
                T_arr[out_idx] = t // STIFF_PERIOD
                Sv_arr[out_idx] = (v_max * t) / abs_var if abs_var > 1e-14 else 1e300
                out_idx += 1

            sigma_prev = sigma

        plt.plot(T_arr, Sv_arr, linewidth=1.5, label=f'N={N}')

    plt.xlabel(f'T (períodos de {STIFF_PERIOD})')
    plt.ylabel(r'$S_v(T)$ (normalizado)')
    plt.title('Stiffness $S_v(T)$ para quantum walk confinado (método analítico)')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig('stiffness_sv_analitico_N101_201_301.png', dpi=150)
    plt.show()

if __name__ == "__main__":
    main()