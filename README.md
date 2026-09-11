# iMSM: Interaction-Based Markov State Models

**iMSM** is a computational framework for constructing Markov state models directly from molecular interaction patterns. Rather than clustering configurations solely on Cartesian coordinates or rigid structural metrics, iMSM categorizes and embeds time-resolved macromolecular interaction topologies to discover functional kinetics and transition networks.

This repository accompanies the manuscript:
> **Learning interpretable kinetic models for biomolecular interaction networks**  
> *Roi Eliasian, Yael Hazan, Timna Tzori, and Barak Raveh*  
> *(Manuscript in preparation / under review)*

---

## General Applicability & Method Overview

iMSMs automate the identification of recurring interaction states and the estimation of their transition networks within a user-specified physical representation.

### When Can iMSM Be Applied?
The approach is generally applicable to **generic molecular dynamics (MD) or Brownian dynamics (BD) simulations** whenever:
- The dynamics of interest can be represented by **recurring local interaction environments** around a chosen **focal entity** $f$.
- The transitions between these interaction environments are **approximately Markovian** at an appropriate lag time $\tau$.

By discretizing time-averaged interaction distributions, iMSMs coarse-grain over microscopic contact configurations that need not share the same local Cartesian geometry, provided they preserve the same statistical interaction pattern around the focal entity. Examples include:
- **Translocation & Crowding**: Characterizing how the interaction environment of a cargo:transport-receptor complex changes as it traverses the central channel of the nuclear pore complex (NPC).
- **Macromolecular Binding & Recognition**: Distinguishing strongly, partially, and weakly engaged states between interacting binding partners or flexible interfaces.
- **Conformational & Folding Dynamics**: Mapping peptide or disordered protein transitions across intramolecular contact topologies (e.g., Chignolin peptide folding).

### 5-Step Workflow
1. **System Partitioning**: The simulated system is partitioned into a set of discrete interacting components, $\mathcal{C} = \{c_1, \ldots, c_N\}$ (e.g., distinct FG Nups, receptor motifs, or residue groups), together with a focal entity $f$ (e.g., a cargo:NTR complex, ligand, or peptide core) that interacts with subsets of these components.
2. **Interaction Trajectory**: Each trajectory is converted into an interaction trajectory $I(t)$ recording contacts between $f$ and $\mathcal{C}$.
3. **Windowed Interaction Histograms**: $I(t)$ is divided into consecutive time windows of duration $\tau$. Interaction statistics in each window are summarized into an interaction histogram $H$ capturing component identities and engagement degree. The histogram is normalized by a fixed reference interaction capacity (to distinguish strongly, partially, and weakly engaged states) and can be augmented with process-specific observables.
4. **Unsupervised Clustering**: Unsupervised clustering of the histograms identifies a reduced set of recurring interaction states, $\mathcal{S} = \{s_1, \ldots, s_k\}$.
5. **Lagged Transition Matrix Inference**: Transitions between states are used to infer the lagged transition-probability matrix:
   $$T_{ij} = P\left(s(t+\tau) = s_j \mid s(t) = s_i\right)$$
   from which stationary distributions, implied timescales, transport observables, and pathway fluxes are estimated.

> **Modeling Choices vs. Inferred States**: The focal entity, interacting components, contact criteria, window duration, and clustering resolution are user modeling choices, whereas state assignments and transition probabilities are inferred directly from the trajectories.

---

## Key Features

- **Modular MSM Pipeline**: End-to-end workflow consisting of 4 clean stages:
  1. `Categorization` — extracts discrete intermolecular or intramolecular interaction signatures across simulation frames.
  2. `Embedding` — projects sparse or high-dimensional interaction topologies into low-dimensional representations.
  3. `Clustering` — identifies kinetically distinct conformational/topological mesostates (including symmetry-aware clustering via `SKM`).
  4. `MSM Estimation` — estimates transition rate generators, implied timescales, stationary distributions, and committor pathways.
- **Nucleocytoplasmic Transport (NPC) Extension**: Specialized tools for analyzing high-throughput nuclear pore complex Brownian dynamics simulations, multi-site FG-repeat sliding, and karyopherin (KAP) translocation.

---

## Quickstart & Tutorials

The best way to get started with `iMSM` is through the interactive tutorials in the [`tutorials/`](tutorials/) directory:

| Tutorial | Description |
| :--- | :--- |
| **[`tutorials/1_chignolin.ipynb`](tutorials/1_chignolin.ipynb)** | **General Biomolecular Dynamics** — Step-by-step walkthrough applying iMSM to the fast-folding Chignolin peptide mini-protein trajectory. |
| **[`tutorials/2_NPC.ipynb`](tutorials/2_NPC.ipynb)** | **Flagship NPC Transport Pipeline** — End-to-end execution of interaction categorization, embedding, clustering, and kinetic graph construction for nuclear transport trajectories. |

---

## Installation

### 1. Clone the repository
```bash
git clone https://github.com/xroi/iMSM.git
cd iMSM
```

### 2. Set up the Conda environment
```bash
# Create and activate environment with core scientific libraries
conda create -n imsm python=3.10 numpy scipy pandas matplotlib seaborn mdtraj -c conda-forge -y
conda activate imsm

# Optional dependencies for specialized plotting and generalized additive models:
conda install -c conda-forge pygam -y
```

### 3. Install iMSM in editable mode
```bash
pip install -e .
```

---

## Reproducing Manuscript Figures

Scripts and helper libraries for reproducing the figures and videos in the manuscript are organized in [`figure_scripts/`](figure_scripts/):

- **Figure 2**: [`figure_scripts/figure_2.py`](figure_scripts/figure_2.py) — 4-site vs. 6-site translocation graphs, stationary distributions, and kinetic comparisons.
- **Figure 3**: [`figure_scripts/figure_3.py`](figure_scripts/figure_3.py) — Multi-variant transport graphs, free-energy grids across pore radii, and divergence metrics.
- **Figure 4**: [`figure_scripts/figure_4.py`](figure_scripts/figure_4.py) — Simulation subset convergence analysis and speedup factor evaluations.
- **Figure 5**: [`figure_scripts/figure_5.py`](figure_scripts/figure_5.py) — Free-energy barriers versus cargo molecular weight across dilated pore diameters.
- **Figure 6**: [`figure_scripts/figure_6.py`](figure_scripts/figure_6.py) — Atomic-scale FG-repeat sliding dynamics, VMD states, and spatial interaction networks.
- **Supplementary Figures**: [`figure_scripts/figure_supp.py`](figure_scripts/figure_supp.py) — Implied timescale validations, Chapman-Kolmogorov tests, and barrier profiles.
- **Video Animations**: [`figure_scripts/figure_videos.py`](figure_scripts/figure_videos.py) — Generation of side-by-side Brownian dynamics simulation and MSM network videos.

---

## Repository Structure

```
.
├── iMSM/              # Core library (categorize, embed, cluster, msm, NPC extensions)
├── tutorials/         # Getting-started Jupyter notebooks (Chignolin folding & NPC)
├── figure_scripts/    # Reproducible scripts and helpers for manuscript figures
├── data/              # Simulation trajectories, transition matrices, and clustering data
├── plots/             # Rendered output figures, panels, and video artifacts
└── pyproject.toml     # Package configuration
```

---

## Citation

```bibtex
@article{eliasian2026imsm,
  title={Learning interpretable kinetic models for biomolecular interaction networks},
  author={Eliasian, Roi and Hazan, Yael and Tzori, Timna and Raveh, Barak},
  journal={In preparation},
  year={2026}
}
```
