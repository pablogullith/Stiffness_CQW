# Stiffness in Confined Quantum Walks — tutorial bundle

Open `Stiffness_in_Confined_Quantum_Walks_Tutorial.ipynb` in Jupyter Notebook or JupyterLab.

## Requirements

- Python 3.9 or newer
- NumPy
- Matplotlib
- SciPy
- Jupyter / JupyterLab

Install missing packages with:

```bash
python -m pip install numpy matplotlib scipy jupyter
```

## Folder contents

- `Stiffness_in_Confined_Quantum_Walks_Tutorial.ipynb`: guided paper explanation, simulator, and plotting code for Figures 1–5.
- `data/`: four precomputed `.npz` caches used to reproduce the full-scale figures without rerunning the long simulations.
- `source_scripts/`: the original supplied Python scripts.
- `Stiffness_in_Confined_Quantum_Walks.pdf`: the supplied paper.

Run the notebook from this extracted folder so its relative `data/` path resolves. Full recomputation of 500,000-step simulations and finite-size sweeps is not run automatically.
