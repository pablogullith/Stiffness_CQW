# Dispersão pelo método analítico
import numpy as np
import matplotlib.pyplot as plt

# Parâmetros
t_max = 1000
Ns = [101, 201, 301]
cores = {101: 'black', 201: 'red', 301: 'blue'}

plt.figure(figsize=(8, 5))

for N in Ns:
    a = (N - 1) // 2
    dim = 2 * N
    posicoes = np.arange(-a, a + 1)

    H = np.array([[1, 1], [1, -1]]) / np.sqrt(2)
    C = np.kron(np.eye(N), H)

    S = np.zeros((dim, dim), dtype=complex)
    for i in range(N):
        if i > 0:
            S[2*(i-1), 2*i] = 1
        else:
            S[2*i+1, 2*i] = 1
        if i < N - 1:
            S[2*(i+1)+1, 2*i+1] = 1
        else:
            S[2*i, 2*i+1] = 1

    U = S @ C

    psi_0 = np.zeros(dim, dtype=complex)
    psi_0[2*a] = 1.0 / np.sqrt(2)
    psi_0[2*a + 1] = 1j / np.sqrt(2)

    X2_op = np.diag(np.repeat(posicoes**2, 2))

    # Decomposição espectral
    evals, evecs = np.linalg.eig(U)
    c_mu = np.conj(evecs).T @ psi_0
    X_munu = np.conj(evecs).T @ X2_op @ evecs

    sigma = np.zeros(t_max)
    for t in range(t_max):
        fases = evals**t
        v_t = c_mu * fases
        M2_t = np.real(np.vdot(v_t, X_munu @ v_t))
        sigma[t] = np.sqrt(abs(M2_t))

    plt.plot(range(t_max), sigma, '-', color=cores[N], label=f'N={N}')

plt.xlim(0, 1000)
plt.ylim(0, 120)
plt.xlabel('Time')
plt.ylabel(r'$\sigma$')
plt.title('Dispersão – Método Analítico')
plt.legend()
plt.tight_layout()
plt.show()