# Stiffness in Confined Quantum Walks

This repository contains the numerical implementation, cached results, generated figures, and guided notebook associated with **“Stiffness in Confined Quantum Walks”** by Pablo G. M. Dantas, Rafael Vieira, and Edgard P. M. Amorim.

The project studies how confinement changes the spreading dynamics of a discrete-time quantum walk. It compares three initial spatial distributions:localized, Gaussian, and truncated uniform; and analyzes the resulting probability distributions, dispersion, coin coherence, spreading stiffness, finite-size scaling, and information divergence.

## About the project

A confined quantum walk evolves on a finite one-dimensional lattice with reflecting auxiliary boundaries. The model combines:

- A Hadamard coin in the bulk lattice
- Pauli-$X$ reflecting coins at the auxiliary boundaries
- A conditional shift of the left- and right-moving coin components
- Localized, Gaussian, and truncated-uniform initial states
- Long-time averages and finite-size simulations

The repository provides both a reproducible implementation and a tutorial path for exploring the analysis behind the manuscript's Figures 1–5.

## Scientific quantities

The scripts and notebook calculate or visualize:

- **Probability density:** $P(j,t)$ across the lattice
- **Dispersion:** $\sigma(t)=\sqrt{\langle j^2\rangle-\langle j\rangle^2}$
- **Coin coherence:** $C(t)=|\langle L|R\rangle|^2$
- **Spreading stiffness:**

  $$
  S_\nu(t)=\frac{\bar{\nu}_H}{\bar{\nu}_H+\nu_\sigma(t)}
  $$
- **Finite-size stiffness moments:** $\langle S_\nu\rangle$ and $\langle S_\nu^2\rangle$
- **Normalized Kullback-Leibler divergence:** $D_{\mathrm{KL}}$

The main conventions are:

- $N=2a+1$ is the number of physical bulk sites.
- $N$ is odd.
- Auxiliary reflecting sites occupy $j=\pm(a+1)$.
- The symmetric coin state is $\lvert\chi\rangle=(\lvert L\rangle+i\lvert R\rangle)/\sqrt{2}$.

## Repository structure

```text
.
├── code/
│   ├── qw_simulation.py
│   ├── fig1_probability_density_and_dispersion.py
│   ├── fig2_quantum_walk_dynamics.py
│   ├── fig3_stiffness_time_evolution.py
│   ├── fig4_finite_size_stiffness.py
│   ├── fig5_kl_divergence.py
│   └── *_cache.npz
├── images/
│   └── Figura01.pdf ... Figura05.pdf
├── Stiffness_in_CQW_Notebook/
│   └── Stiffness_in_CQW_Notebook/
│       ├── Stiffness_CQW_Notebook.ipynb
│       ├── data/
│       ├── source_scripts/
│       └── README.md
└── Stiffness_in_Confined_Quantum_Walks.pdf
```

### Figure scripts

| Script | Output |
|---|---|
| [`fig1_probability_density_and_dispersion.py`](code/fig1_probability_density_and_dispersion.py) | Analytical probability density and dispersion |
| [`fig2_quantum_walk_dynamics.py`](code/fig2_quantum_walk_dynamics.py) | Probability distributions, dispersion, and coin coherence |
| [`fig3_stiffness_time_evolution.py`](code/fig3_stiffness_time_evolution.py) | Spreading stiffness over time |
| [`fig4_finite_size_stiffness.py`](code/fig4_finite_size_stiffness.py) | Finite-size stiffness dependence |
| [`fig5_kl_divergence.py`](code/fig5_kl_divergence.py) | Normalized KL divergence |

The generated PDFs are saved in [`images/`](images/).

## Getting started

### Requirements

Python 3.9 or newer is recommended:

- NumPy
- SciPy
- Matplotlib
- Numba
- Jupyter Notebook or JupyterLab

Create an environment and install the dependencies:

```bash
python -m venv .venv
```

Activate the environment:

- Windows: `.venv\Scripts\activate`
- Linux/macOS: `source .venv/bin/activate`

Then install the packages:

```bash
python -m pip install numpy scipy matplotlib numba jupyter
```

### Run the guided notebook

Change to the notebook directory and start Jupyter:

```bash
cd Stiffness_in_CQW_Notebook/Stiffness_in_CQW_Notebook
jupyter notebook
```

Open [`Stiffness_CQW_Notebook.ipynb`](Stiffness_in_CQW_Notebook/Stiffness_in_CQW_Notebook/Stiffness_CQW_Notebook.ipynb). It includes the paper's theoretical context, numerical examples, and cache-backed plots.

Run the notebook from its own directory so relative paths to [`data/`](Stiffness_in_CQW_Notebook/Stiffness_in_CQW_Notebook/data/) resolve correctly.

### Generate figures

Run a figure script from the repository root:

```bash
python code/fig1_probability_density_and_dispersion.py
python code/fig2_quantum_walk_dynamics.py
python code/fig3_stiffness_time_evolution.py
python code/fig4_finite_size_stiffness.py
python code/fig5_kl_divergence.py
```

The scripts load supplied caches when available and save each result as `images/Figura0X.pdf`.

## Precomputed data

The repository includes compressed NumPy caches for its most expensive calculations:

- `code/qw_simulation_cache.npz`
- `code/fig3_cache.npz`
- `code/fig4_cache.npz`
- `code/fig5_cache.npz`

The notebook bundle contains copies of these caches in [`Stiffness_in_CQW_Notebook/Stiffness_in_CQW_Notebook/data/`](Stiffness_in_CQW_Notebook/Stiffness_in_CQW_Notebook/data/).

Using the caches allows the full-scale figures to be regenerated quickly. Some loaders expose `force_recompute=True` for users who want to repeat the underlying simulations.

Full recomputation may involve up to 500,000 time steps, transient cutoffs of 20,000–100,000 steps, and a finite-size sweep from $N=101$ to $N=1001$. It can take several minutes and depends on CPU performance and Numba compilation.

## Included manuscript

The full paper is available in [`Stiffness_in_Confined_Quantum_Walks.pdf`](Stiffness_in_Confined_Quantum_Walks.pdf).

## Citation

Please cite:

P. G. M. Dantas, R. Vieira, and E. P. M. Amorim, **“Stiffness in Confined Quantum Walks.”**
