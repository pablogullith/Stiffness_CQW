import numpy as np
import matplotlib.pyplot as plt

def simulate_KL_divergence(N, max_steps):
    a = (N - 1) // 2
    positions = np.arange(-a, a + 1)
    
    # Initialize quantum state array at the origin
    psi = np.zeros((2, N), dtype=complex)
    
    # Apply initial localized coin state (alpha = pi/2, beta = pi/2)
    psi[0, a] = 1 / np.sqrt(2)
    psi[1, a] = 1j / np.sqrt(2)
    
    # Hadamard coin operator
    H = np.array([[1, 1], [1, -1]]) / np.sqrt(2)
    
    # Track instantaneous probabilities
    prob_t = np.sum(np.abs(psi)**2, axis=0) # P(j, 0)
    
    D_KL_avg = []
    sum_D_KL = 0.0
    
    for t in range(max_steps):
        # Apply coin operator
        psi_L = H[0, 0] * psi[0] + H[0, 1] * psi[1]
        psi_R = H[1, 0] * psi[0] + H[1, 1] * psi[1]
        
        # Apply conditional shift operator with reflecting boundaries
        psi_new = np.zeros((2, N), dtype=complex)
        psi_new[0, :-1] = psi_L[1:]  # Move left
        psi_new[1, 1:] = psi_R[:-1]  # Move right
        
        # NOT-gate reflections at the endpoints
        psi_new[1, 0] += psi_L[0]
        psi_new[0, -1] += psi_R[-1]
        
        prob_next = np.sum(np.abs(psi_new)**2, axis=0) # P(j, t+1)
        
        # Calculate accessible sites based on parity rule: j === t (mod 2)
        M_t = len([j for j in positions if abs(j) <= min(t, a) and (j % 2) == (t % 2)])
        M_t_plus_1 = len([j for j in positions if abs(j) <= min(t + 1, a) and (j % 2) == ((t + 1) % 2)])
        M2_t = M_t + M_t_plus_1
        
        # Two-step normalized probability distribution P^(2)(j,t)
        P_2 = (M_t * prob_t + M_t_plus_1 * prob_next) / M2_t
        
        # Calculate Shannon entropy (excluding zero probabilities)
        P_2_nonzero = P_2[P_2 > 1e-15]
        H_B = -np.sum(P_2_nonzero * np.log(P_2_nonzero))
        
        # Calculate normalized Kullback-Leibler divergence
        if M2_t > 1:
            D_KL = 1.0 - (H_B / np.log(M2_t))
        else:
            D_KL = 0.0
            
        # Update running cumulative average for the plot
        sum_D_KL += D_KL
        D_KL_avg.append(sum_D_KL / (t + 1))
        
        # Step forward in time
        psi = psi_new
        prob_t = prob_next
        
    return D_KL_avg

# Parameters mimicking the Figure 4 setup
# Note: For rapid execution, max_steps is bounded to 1000. 
# To observe the extreme long-time multi-period plateau saturation exactly as 
# plotted in the article, max_steps should be increased into the tens of thousands.
max_steps = 1630000 
lattice_sizes = [101, 201, 301]
colors = ['black', 'red', 'blue']

plt.figure(figsize=(10, 6))

for N, color in zip(lattice_sizes, colors):
    dkl_average = simulate_KL_divergence(N, max_steps)
    plt.plot(range(1, max_steps + 1), dkl_average, color=color, label=f'N = {N}')

plt.xlabel('Time steps $t$')
plt.ylabel('Normalized Two-Step KL Divergence $\overline{\mathcal{D}}_{KL}^N$')
plt.title('Normalized KL Distance from Uniform Distribution')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.6)
plt.savefig('DKL_plot.png', dpi=300, bbox_inches='tight')
plt.show()