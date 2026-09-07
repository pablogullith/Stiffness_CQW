import numpy as np
import matplotlib.pyplot as plt

def simulate_stiffness(N, max_steps):
    # Determine boundary limits -a and a from the total number of sites N
    a = (N - 1) // 2
    positions = np.arange(-a, a + 1)
    
    # Initialize quantum state array: index 0 for |L>, index 1 for |R>
    psi = np.zeros((2, N), dtype=complex)
    
    # Apply initial localized coin state (alpha = pi/2, beta = pi/2)
    psi[0, a] = 1 / np.sqrt(2)
    psi[1, a] = 1j / np.sqrt(2)
    
    # Define Hadamard coin operator
    H = np.array([[1, 1], [1, -1]]) / np.sqrt(2)
    
    # Calculate the asymptotic spreading rate of the free Hadamard walk
    nu_H = np.sqrt(1 - 1/np.sqrt(2))
    
    # Tracking variables for stiffness equations
    sigma_prev = 0.0 
    running_sum = 0.0
    S_nu = []
    
    for t in range(1, max_steps + 1):
        # 1. Apply coin operator
        psi_L = H[0, 0] * psi[0] + H[0, 1] * psi[1]
        psi_R = H[1, 0] * psi[0] + H[1, 1] * psi[1]
        
        # 2. Conditional shift operator with reflecting boundaries
        psi_new = np.zeros((2, N), dtype=complex)
        
        # Bulk movement
        psi_new[0, :-1] = psi_L[1:] # Move left
        psi_new[1, 1:] = psi_R[:-1] # Move right
        
        # Reflecting boundaries via NOT gates
        psi_new[1, 0] += psi_L[0]   # Reverses left-moving to right-moving at -a
        psi_new[0, -1] += psi_R[-1] # Reverses right-moving to left-moving at a
        
        psi = psi_new
        
        # Calculate standard deviation sigma(t)
        prob = np.abs(psi[0])**2 + np.abs(psi[1])**2
        mean_x = np.sum(positions * prob)
        mean_x2 = np.sum(positions**2 * prob)
        sigma_current = np.sqrt(np.maximum(mean_x2 - mean_x**2, 0))
        
        # Calculate mean spreading activity over W steps
        running_sum += np.abs(sigma_current - sigma_prev)
        nu_sigma_bar = running_sum / t
        
        # Calculate dimensionless spreading stiffness
        S_nu.append(nu_H / (nu_H + nu_sigma_bar))
        sigma_prev = sigma_current
        
    return S_nu

# Replicate Figure 3 for confinement sizes N = 101, 201, 301
plt.figure(figsize=(10, 6))
max_t = 163000

for N in [101, 201, 301]:
    stiffness_vals = simulate_stiffness(N, max_t)
    plt.plot(range(1, max_t + 1), stiffness_vals, label=f'N = {N}')

plt.xlabel('Observation window (W)')
plt.ylabel('Stiffness S_nu')
plt.title('Spreading Stiffness of Confined Quantum Walks')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.7)
plt.show()