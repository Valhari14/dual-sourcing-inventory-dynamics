# Cyclic Dual-Sourcing Controllers

This module implements multi-period (cyclic) controllers for dual-sourcing inventory problems. Instead of making one decision per period, a cyclic controller makes a single forward pass per **cycle** of `n_cycles` periods, determining all order quantities for the entire cycle at once.

## Contents

| File | Description |
|---|---|
| `base.py` | Abstract base classes (`BaseDPController`, `BaseNeuralController`) |
| `cyclic_dual_neural.py` | `CyclicDualNeuralController` — neural network cyclic policy |
| `dynamic_programming.py` | `DynamicProgrammingController` — optimal cyclic policy via value iteration |
| `dynamic_programming_parity.py` | Experimental parity-constrained variant (unused) |

---

## Cyclic Decision Structure

For a cycle of length `n_cycles = 2`:

| Period in cycle | Regular order | Expedited order |
|---|---|---|
| 0 | `q_r_0` | `q_e_0` |
| 1 | — | `q_e_1` |

The network outputs `n_cycles + 1` values: `(q_r_0, q_e_0, q_e_1, ...)`. The regular order is placed only once (period 0); expedited orders are placed in every period of the cycle.

---

## `CyclicDualNeuralController`

### Constructor

```python
from idinn.cyclic_dual_controller import CyclicDualNeuralController
import torch

controller = CyclicDualNeuralController(
    hidden_layers=[64, 32, 16, 8],   # hidden layer sizes
    activation=torch.nn.CELU(alpha=1.0),
    n_cycles=2,                       # cycle length (must be > 1)
)
```

| Parameter | Default | Description |
|---|---|---|
| `hidden_layers` | `[128, 64, 32, 16, 8]` | Number of neurons per hidden layer |
| `activation` | `CELU(alpha=1.0)` | Activation function between layers |
| `n_cycles` | `2` | Number of periods per cycle |

> `n_cycles` must be greater than 1.

### Network Architecture

- **Input:** `regular_lead_time + expedited_lead_time + 1` features (current inventory + pipeline state)
- **Hidden:** configurable via `hidden_layers`
- **Output:** `n_cycles + 1` units with ReLU, clamped to `[0, 20]`

### Training

```python
from idinn.sourcing_model import DualSourcingModel
from idinn.demand import UniformDemand

sourcing_model = DualSourcingModel(
    regular_lead_time=2,
    expedited_lead_time=0,
    regular_order_cost=0,
    expedited_order_cost=20,
    holding_cost=5,
    shortage_cost=495,
    init_inventory=6,
    demand_generator=UniformDemand(low=0, high=4),
    batch_size=1,
)

controller.fit(
    sourcing_model=sourcing_model,
    sourcing_periods=100,
    epochs=8500,
    validation_sourcing_periods=1000,
    validation_freq=50,
    log_freq=10,
    init_inventory_lr=1e-1,
    init_inventory_freq=4,
    parameters_lr=3e-4,
    seed=42,
    checkpoint_path="models/trained/finetuned_lt4_nc2_sc95_ec20_d0-4.pt",  # optional; saves best by val cost
)
```

| Parameter | Default | Description |
|---|---|---|
| `sourcing_periods` | — | Periods per training epoch (scaled by `n_cycles` internally) |
| `epochs` | — | Number of training epochs |
| `validation_sourcing_periods` | `1000` | Periods for validation evaluation |
| `validation_freq` | `50` | Evaluate validation every N epochs |
| `log_freq` | `10` | Log training cost every N epochs |
| `init_inventory_lr` | `1e-1` | Learning rate for `init_inventory` optimizer |
| `init_inventory_freq` | `4` | Update `init_inventory` every N epochs (else update NN weights) |
| `parameters_lr` | `1e-4` | Learning rate for NN parameters |
| `seed` | `None` | Random seed for reproducibility |
| `checkpoint_path` | `None` | If set, saves the best checkpoint here during training |

**Optimizer:** Adam with cosine annealing LR schedule and gradient clipping (`max_norm=1.0`).

### Inference

```python
orders = controller.predict(
    current_inventory=6,
    past_regular_orders=[2, 3],
    past_expedited_orders=[1],
)
# returns (q_r_0, q_e_0, q_e_1, ...) as a tuple of tensors
```

### Evaluation

```python
import torch

costs = []
with torch.no_grad():
    for seed in range(100):
        cost = controller.get_average_cost(
            sourcing_model=sourcing_model,
            sourcing_periods=1000,
            seed=seed,
        )
        costs.append(cost)

mean_cost = torch.mean(torch.stack(costs)).item()
std_cost  = torch.std(torch.stack(costs)).item()
```

---

## Checkpoint Management

### Saving

Checkpoints are saved automatically during `fit()` whenever validation cost improves (if `checkpoint_path` is provided). To save manually:

```python
controller.save_checkpoint("models/trained/finetuned_lt4_nc2_sc95_ec20_d0-4.pt")
```

The checkpoint dict contains: `model_state_dict`, `hidden_layers`, `n_cycles`, `init_inventory`.

### Loading

```python
sourcing_model = DualSourcingModel(...)   # must match training config

controller = CyclicDualNeuralController.load_checkpoint(
    path="models/trained/finetuned_lt4_nc2_sc95_ec20_d0-4.pt",
    sourcing_model=sourcing_model,
)
```

`load_checkpoint` restores `hidden_layers`, `n_cycles`, and `init_inventory` from the file. The `sourcing_model` must have the same `regular_lead_time` and `expedited_lead_time` as during training.

> **Backward compatibility:** Checkpoints saved before `n_cycles` was added to the checkpoint format will default to `n_cycles=2` on load.

---

## Neural Network Training & Evaluation CLI

`src/idinn/finetuning/hp_grid_search.py` provides an end-to-end command-line interface for training, transfer fine-tuning, and evaluating `CyclicDualNeuralController`.

```bash
# 1. Train base model from scratch:
python src/idinn/finetuning/hp_grid_search.py \
    --mode train --n_cycles 3 --lt_s 2 --shortage_cost 95 --demand_high 4 \
    --checkpoint_dir models/base_c3_ls2_b95

# 2. Transfer fine-tuning (lead-time expansion):
python src/idinn/finetuning/hp_grid_search.py \
    --mode train --n_cycles 3 --lt_s 3 --shortage_cost 95 --demand_high 4 \
    --lr 0.0002 --base_checkpoint models/base_c3_ls2_b95 \
    --checkpoint_dir models/transfer_c3_ls3_b95

# 3. Policy evaluation & GAP% computation over 500 test seeds:
python src/idinn/finetuning/hp_grid_search.py \
    --mode infer --n_cycles 3 --lt_s 3 --shortage_cost 95 --demand_high 4 \
    --vf 106.0125 --checkpoint_dir models/transfer_c3_ls3_b95
```

---

## `DynamicProgrammingController` (cyclic)

Computes the optimal cyclic policy via value iteration using Numba JIT. Supports cycle lengths 1, 2, and 3. Requires `UniformDemand` and `expedited_lead_time=0`.

```python
from idinn.cyclic_dual_controller import DynamicProgrammingController

controller = DynamicProgrammingController(cycle_length=2)
controller.fit(sourcing_model=sourcing_model, max_iterations=1000000, tolerance=1e-7)

orders = controller.predict(
    current_inventory=6,
    past_regular_orders=[2, 3],
    past_expedited_orders=[1],
)
```
