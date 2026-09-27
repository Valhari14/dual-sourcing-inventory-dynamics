# Dual-Sourcing Inventory Dynamics

This repository extends the **IDINN (Inventory-Dynamics Control with Neural Networks)** framework with **exact Dynamic Programming (DP) bounds** and **deep Recurrent Neural Network (RNN) controllers** for periodic and unrestricted dual-sourcing inventory systems.

> **Acknowledgement**
>
> This repository builds upon the foundational **IDINN** architecture originally developed by the Computational Science group:
> https://gitlab.com/ComputationalScience/idinn

---

## Overview

In dual-sourcing inventory management, an inventory manager balances two supply channels:
1. **Regular Supplier:** Lower unit acquisition cost, longer lead time ($\ell_s \ge 2$).
2. **Expedited Supplier:** Higher unit cost, short or immediate delivery ($\ell_e = 0$).

This framework addresses two core problem formulations:
* **Problem I (Unrestricted Dual-Sourcing):** Regular orders can be placed in every discrete time period. Solved to optimality using multi-dimensional dynamic programming (`run_dp_programming.py`).
* **Problem II (Periodic / Cyclic Dual-Sourcing):** Regular orders are restricted to cycle boundaries $t \equiv 0 \pmod N$, while expedited replenishment remains available each period. Certified optimal policy bounds are computed via cyclic dynamic programming (`run_dp_bound.py`), and near-optimal closed-loop control is learned using cyclic neural network controllers (`src/idinn/finetuning/hp_grid_search.py`).

---

## Installation

Clone the repository:
```bash
git clone https://github.com/Valhari14/dual-sourcing-inventory-dynamics.git
cd dual-sourcing-inventory-dynamics
```

Install editable package and dependencies:
```bash
pip install -e .
```
or
```bash
pip install -r requirements.txt
```

---

## Repository Structure

```
├── src/
│   └── idinn/
│       ├── cyclic_dual_controller/  # Cyclic DP bound & neural controllers (Problem II)
│       ├── dual_controller/         # Unrestricted DP solvers & bounds (Problem I)
│       ├── finetuning/              # Unified training & transfer pipeline (hp_grid_search.py)
│       ├── sourcing_model.py        # Dual-sourcing environment simulation dynamics
│       └── demand.py                # Stochastic demand distributions (UniformDemand)
├── docs/                            # Documentation
├── tests/                           # Unit tests
├── app/                             # Web visualizer application
├── run_dp_programming.py            # Problem I DP benchmark runner
└── run_dp_bound.py                  # Problem II Cyclic DP bound runner
```

> **Note on Model Checkpoints:** Checkpoint directories (`models/`, `checkpoints/`) containing `.pt` binary weights are intentionally excluded from version control to maintain a lightweight repository. Models are generated on-demand during training or transfer fine-tuning.

---

## Execution Guide

### 1. Dynamic Programming Benchmarks
Compute certified DP cost baselines:

```bash
# Problem I (Unrestricted dual-sourcing DP benchmark)
python run_dp_programming.py --cycle_length 2 --regular_lead_time 2 --backlog_cost 495 --demand_max 4

# Problem II (Periodic dual-sourcing cyclic DP bound)
python run_dp_bound.py --cycle_length 3 --regular_lead_time 2 --backlog_cost 95 --demand_max 4
```

### 2. Neural Controller Training & Transfer Learning
Train and evaluate neural controllers via the unified CLI (`src/idinn/finetuning/hp_grid_search.py`):

```bash
# 1. Train base policy from scratch (lr=0.001)
python src/idinn/finetuning/hp_grid_search.py \
    --mode train \
    --n_cycles 3 --lt_s 2 --shortage_cost 95 --demand_high 4 \
    --checkpoint_dir models/t2_c3_d4_b95_ls2 \
    --device cuda

# 2. Penalty Transfer Learning (b=95 -> b=495, lr=0.0002)
python src/idinn/finetuning/hp_grid_search.py \
    --mode train \
    --n_cycles 3 --lt_s 2 --shortage_cost 495 --demand_high 4 \
    --lr 0.0002 \
    --base_checkpoint models/t2_c3_d4_b95_ls2 \
    --checkpoint_dir models/t2_c3_d4_b495_ls2 \
    --device cuda

# 3. Lead-Time Expansion Transfer (ls=2 -> ls=3 via zero-padding, lr=0.0002)
python src/idinn/finetuning/hp_grid_search.py \
    --mode train \
    --n_cycles 3 --lt_s 3 --shortage_cost 95 --demand_high 4 \
    --lr 0.0002 \
    --base_checkpoint models/t2_c3_d4_b95_ls2 \
    --checkpoint_dir models/t2_c3_d4_b95_ls3 \
    --device cuda

# 4. Evaluation and Certified Optimality GAP% (500 Monte Carlo seeds)
python src/idinn/finetuning/hp_grid_search.py \
    --mode infer \
    --n_cycles 3 --lt_s 2 --shortage_cost 495 --demand_high 4 \
    --vf 111.7282 \
    --checkpoint_dir models/t2_c3_d4_b495_ls2 \
    --device cuda
```

---

## Citations

If you build upon this work or use the original IDINN framework, please cite:

- Böttcher, L., Asikis, T., & Fragkos, I. (2023). *Control of Dual-Sourcing Inventory Systems using Recurrent Neural Networks*. INFORMS Journal on Computing.
- Li, J., Asikis, T., Fragkos, I., & Böttcher, L. (2025). *idinn: A Python package for inventory-dynamics control with neural networks*. Journal of Open Source Software.

---

## License

Please refer to the licensing terms of the original IDINN project for the underlying framework.

