# Dual-Sourcing Inventory Dynamics

This repository contains an **ongoing experimental version** of the **IDINN (Inventory-Dynamics Control with Neural Networks)** framework. It extends the original project with **dynamic programming methods for periodic dual-sourcing inventory systems**.

> **Acknowledgement**
>
> This repository is based on the original **IDINN** project developed by the Computational Science group:
> https://gitlab.com/ComputationalScience/idinn

## Overview

The repository includes implementations for:

- Dynamic programming controllers for dual-sourcing inventory systems
- Parity-based dynamic programming algorithms
- Neural network controllers from the original IDINN framework
- Customizable inventory and demand models
- Benchmarking and experimentation utilities

The project is intended for research, experimentation, and further development of inventory optimization methods.

## Installation

Clone the repository

```bash
git clone https://github.com/Valhari14/dual-sourcing-inventory-dynamics.git
cd dual-sourcing-inventory-dynamics
```

Install the package

```bash
pip install -e .
```

or install the dependencies

```bash
pip install -r requirements.txt
```

## Repository Structure

```
src/                    # Source code
docs/                   # Documentation
tests/                  # Unit tests
app/                    # Demo application

run_dp_programming.py   # Dynamic programming experiments
run_dp_bound.py         # DP bound computations
```

## Original IDINN Project

For the original implementation and documentation, see:

- GitLab: https://gitlab.com/ComputationalScience/idinn
- Documentation: https://inventory-optimization.readthedocs.io

## Citation

If you use the original IDINN framework in your research, please cite:

- Böttcher, L., Asikis, T., & Fragkos, I. (2023). *Control of Dual-Sourcing Inventory Systems using Recurrent Neural Networks*. INFORMS Journal on Computing.
- Li, J., Asikis, T., Fragkos, I., & Böttcher, L. (2025). *idinn: A Python package for inventory-dynamics control with neural networks*. Journal of Open Source Software.

If you use the dynamic programming extensions developed in this repository, please cite the corresponding publication once available.

## License

Please refer to the licensing terms of the original IDINN project for the underlying framework.
