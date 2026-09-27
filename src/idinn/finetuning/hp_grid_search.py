"""
hp_grid_search.py -- Neural Network Training & Transfer Learning for Cyclic Dual-Sourcing

Provides end-to-end training, transfer learning, and evaluation for CyclicDualNeuralController
under periodic dual-sourcing inventory dynamics (Böttcher et al., 2023).

Usage Examples:
  # 1. Train base model from scratch:
  python src/idinn/finetuning/hp_grid_search.py \\
      --mode train --n_cycles 3 --lt_s 2 --shortage_cost 95 --demand_high 4 \\
      --checkpoint_dir models/base_c3_ls2_b95

  # 2. Transfer fine-tuning (e.g., lead-time expansion from ls=2 to ls=3):
  python src/idinn/finetuning/hp_grid_search.py \\
      --mode train --n_cycles 3 --lt_s 3 --shortage_cost 95 --demand_high 4 \\
      --lr 0.0002 --base_checkpoint models/base_c3_ls2_b95 \\
      --checkpoint_dir models/transfer_c3_ls3_b95

  # 3. Evaluate trained policy across 500 test seeds and compute GAP%:
  python src/idinn/finetuning/hp_grid_search.py \\
      --mode infer --n_cycles 3 --lt_s 3 --shortage_cost 95 --demand_high 4 \\
      --vf 106.0125 --checkpoint_dir models/transfer_c3_ls3_b95
"""

import argparse
import concurrent.futures
import logging
import math
import multiprocessing as mp
import os
import shutil
import sys
import time
from typing import List, Optional

import torch
from tqdm import tqdm

from src.idinn.cyclic_dual_controller.cyclic_dual_neural import CyclicDualNeuralController
from src.idinn.demand import UniformDemand
from src.idinn.sourcing_model import DualSourcingModel

# -----------------------------------------------------------------------------
# Dual-Sourcing Benchmark Constants (Böttcher et al., 2023)
# -----------------------------------------------------------------------------
HOLDING_COST: float = 5.0
EXPEDITED_COST: float = 20.0
REGULAR_ORDER_COST: float = 0.0
EXPEDITED_LEAD_TIME: int = 0
DEMAND_LOW: int = 0

# Optimization & Architecture Defaults
DEFAULT_HIDDEN_LAYERS: List[int] = [64, 32, 16, 8, 4]
BATCH_SIZE: int = 512
INIT_INVENTORY_LR: float = 0.1
DEFAULT_EPOCHS: int = 6000
DEFAULT_PATIENCE: int = 15
OPTIMIZER_TYPE: str = "rmsprop"
USE_GRAD_CLIP: bool = True

# Evaluation Constants
EVAL_PERIODS: int = 1000
EVAL_SEEDS: int = 500
VF_GAP_TOL: float = 0.005
VF_PATIENCE: int = 5

DEFAULT_DEVICE: str = "cuda" if torch.cuda.is_available() else "cpu"

GRID_LRS: List[float] = [3e-4, 1e-3, 3e-3]
GRID_LAYERS: List[List[int]] = [
    [64, 32, 16, 8, 4],
    [128, 64, 32, 16, 8, 4, 2],
]

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)


def _sourcing_periods_for(lt_s: int) -> int:
    """Horizon scaling per regular lead time: T=100/150/200 for lt_s=2/3/4+."""
    if lt_s <= 2:
        return 100
    elif lt_s == 3:
        return 150
    elif lt_s == 4:
        return 200
    else:
        return 200 + (lt_s - 4) * 50


def _build_sourcing_model(
    lt_s: int,
    n_cycles: int,
    shortage_cost: float,
    demand_high: int,
    expedited_cost: float = EXPEDITED_COST,
    demand_low: int = DEMAND_LOW,
    batch_size: int = BATCH_SIZE,
    init_inventory: Optional[float] = None,
) -> DualSourcingModel:
    """Instantiate the dual-sourcing simulation environment with benchmark parameters."""
    if init_inventory is None:
        mean_demand = (demand_low + demand_high) / 2.0
        init_inv = float((lt_s + 1) * mean_demand)
    else:
        init_inv = float(init_inventory)

    return DualSourcingModel(
        regular_lead_time=lt_s,
        expedited_lead_time=EXPEDITED_LEAD_TIME,
        regular_order_cost=REGULAR_ORDER_COST,
        expedited_order_cost=expedited_cost,
        holding_cost=HOLDING_COST,
        shortage_cost=shortage_cost,
        init_inventory=init_inv,
        demand_generator=UniformDemand(demand_low, demand_high),
        batch_size=batch_size,
    )


def _load_pretrained(base_checkpoint: str, hidden_layers: List[int], n_cycles: int, sourcing_model: DualSourcingModel):
    """
    Load pre-trained weights and adapt to new lead time or cycle length.
    Expands input dimension by preserving existing pipeline weights and zero-initializing
    new lead-time slots. Re-initializes output heads if cycle length N changes.
    """
    if os.path.isdir(base_checkpoint):
        best_pt = os.path.join(base_checkpoint, "best_model.pt")
        if os.path.exists(best_pt):
            base_checkpoint = best_pt

    if not os.path.exists(base_checkpoint):
        raise FileNotFoundError(f"Base checkpoint not found: {base_checkpoint!r}.")

    ckpt = torch.load(base_checkpoint, map_location="cpu")
    ps = ckpt["model_state_dict"]
    ph = ckpt.get("hidden_layers", hidden_layers)
    pc = ckpt.get("n_cycles", n_cycles)

    if ph != hidden_layers:
        logger.info("Inheriting base checkpoint hidden_layers: %s (requested: %s)", ph, hidden_layers)
        hidden_layers = ph

    ctrl = CyclicDualNeuralController(hidden_layers=hidden_layers, n_cycles=n_cycles)
    ctrl.init_layers(
        regular_lead_time=sourcing_model.get_regular_lead_time(),
        expedited_lead_time=sourcing_model.get_expedited_lead_time(),
    )
    ctrl.sourcing_model = sourcing_model
    sourcing_model.init_inventory.data.fill_(ckpt["init_inventory"])

    cs = ctrl.state_dict()
    skip = set()

    pi = ps["model.0.weight"].shape[1]
    ci = cs["model.0.weight"].shape[1]

    if pi != ci:
        if ci > pi:
            # Lead-time expansion: copy trained weights for existing pipeline slots, zero-init new slots
            with torch.no_grad():
                new_w = cs["model.0.weight"].clone()
                new_w[:, :pi] = ps["model.0.weight"]
                new_w[:, pi:] = 0.0
                cs["model.0.weight"] = new_w
                cs["model.0.bias"] = ps["model.0.bias"].clone()
            skip.update({"model.0.weight", "model.0.bias"})
            logger.info("Input dim expanded %d -> %d: preserved weights, zero-initialized %d new slot(s).", pi, ci, ci - pi)
        else:
            skip.update({"model.0.weight", "model.0.bias"})
            logger.info("Input dim reduced %d -> %d: re-initializing input layer.", pi, ci)

    ok = f"model.{2 * len(ph)}"
    po = ps[f"{ok}.weight"].shape[0]
    co = cs[f"{ok}.weight"].shape[0]
    if po != co:
        skip.update({f"{ok}.weight", f"{ok}.bias"})
        logger.info("Output dim %d -> %d (cycles %d -> %d): re-initializing output heads.", po, co, pc, n_cycles)

    if skip:
        compat = {k: v for k, v in ps.items() if k not in skip}
        cs.update(compat)
        ctrl.load_state_dict(cs)
    else:
        ctrl.load_state_dict(ps)
        logger.info("All model dimensions match -- loaded full pre-trained checkpoint.")

    return ctrl


def _train_one(
    *,
    hidden_layers: List[int],
    n_cycles: int,
    sourcing_model: DualSourcingModel,
    sourcing_periods: int,
    epochs: int,
    parameters_lr: float,
    init_inventory_lr: float,
    seed: int,
    checkpoint_path: str,
    base_checkpoint: Optional[str],
    device: str,
    patience: int = DEFAULT_PATIENCE,
    use_scheduler: bool = False,
    target_vf: Optional[float] = None,
    vf_gap_tol: float = VF_GAP_TOL,
    vf_patience: int = VF_PATIENCE,
):
    """Execute training for a single model instance."""
    if base_checkpoint:
        ctrl = _load_pretrained(base_checkpoint, hidden_layers, n_cycles, sourcing_model)
    else:
        ctrl = CyclicDualNeuralController(hidden_layers=hidden_layers, n_cycles=n_cycles)

    ctrl.fit(
        sourcing_model=sourcing_model,
        sourcing_periods=sourcing_periods,
        epochs=epochs,
        validation_sourcing_periods=1000,
        validation_freq=min(200, epochs),
        log_freq=10,
        init_inventory_lr=init_inventory_lr,
        parameters_lr=parameters_lr,
        seed=seed,
        checkpoint_path=checkpoint_path,
        optimizer_type=OPTIMIZER_TYPE,
        use_scheduler=use_scheduler,
        use_grad_clip=USE_GRAD_CLIP,
        device=device,
        patience=patience,
        target_vf=target_vf,
        vf_gap_tol=vf_gap_tol,
        vf_patience=vf_patience,
    )


def run_scan(args):
    """Execute hyperparameter grid scan across candidate learning rates and layer sizes."""
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    sp = args.sourcing_periods or _sourcing_periods_for(args.lt_s)
    epochs = args.epochs or 800

    logger.info("Starting HP Scan: lt_s=%d, N=%d, b=%.1f, epochs=%d, device=%s",
                args.lt_s, args.n_cycles, args.shortage_cost, epochs, args.device)

    results = []
    total = len(GRID_LRS) * len(GRID_LAYERS)
    c = 0

    for lr in GRID_LRS:
        for layers in GRID_LAYERS:
            c += 1
            ls = ",".join(str(x) for x in layers)
            ckpt = os.path.join(args.checkpoint_dir, f"scan_lr{lr}_layers{ls.replace(',', '_')}.pt")
            print(f"\n[{c}/{total}] lr={lr} layers=[{ls}]")
            t0 = time.time()

            sm = _build_sourcing_model(
                args.lt_s, args.n_cycles, args.shortage_cost,
                demand_high=args.demand_high, expedited_cost=args.expedited_cost,
                demand_low=args.demand_low, batch_size=args.batch_size,
                init_inventory=args.init_inventory,
            )
            _train_one(
                hidden_layers=layers, n_cycles=args.n_cycles, sourcing_model=sm,
                sourcing_periods=sp, epochs=epochs, parameters_lr=lr,
                init_inventory_lr=args.init_inventory_lr, seed=args.seed,
                checkpoint_path=ckpt, base_checkpoint=args.base_checkpoint,
                device=args.device, patience=args.patience, use_scheduler=args.use_scheduler,
            )
            elapsed = time.time() - t0

            esm = _build_sourcing_model(
                args.lt_s, args.n_cycles, args.shortage_cost,
                demand_high=args.demand_high, expedited_cost=args.expedited_cost,
                demand_low=args.demand_low, batch_size=args.batch_size,
                init_inventory=args.init_inventory,
            )
            ec = CyclicDualNeuralController.load_checkpoint(ckpt, esm, device=args.device)
            with torch.no_grad():
                ev = [ec.get_average_cost(esm, 1000, seed=s) for s in range(20)]
            mu = torch.stack(ev).mean().item()
            std = torch.stack(ev).std().item()
            results.append((mu, lr, layers, std, ckpt))
            print(f"  {elapsed / 60:.1f}min | val_mean={mu:.4f} std={std:.4f}")

    results.sort(key=lambda x: x[0])
    bmu, blr, bl, bstd, bckpt = results[0]
    dest = os.path.join(args.checkpoint_dir, "best_scan_model.pt")
    shutil.copy(bckpt, dest)
    print(f"\nWinner -> {dest} | lr={blr} layers={bl} val_mean={bmu:.4f}")


def _seed_worker(worker_cfg: dict) -> dict:
    """Worker task for multiprocessing seed training."""
    sys.path.insert(0, os.getcwd())

    seed = worker_cfg["seed"]
    ckpt_path = worker_cfg["ckpt_path"]

    seed_log_dir = os.path.dirname(ckpt_path)
    os.makedirs(seed_log_dir, exist_ok=True)
    seed_log_path = os.path.join(seed_log_dir, f"seed_{seed}.log")

    root_logger = logging.getLogger()
    root_logger.handlers.clear()
    root_logger.setLevel(logging.INFO)
    fh = logging.FileHandler(seed_log_path, mode="w")
    fh.setLevel(logging.INFO)
    fh.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s"))
    root_logger.addHandler(fh)

    sm = _build_sourcing_model(
        worker_cfg["lt_s"], worker_cfg["n_cycles"], worker_cfg["shortage_cost"],
        demand_high=worker_cfg["demand_high"], expedited_cost=worker_cfg["expedited_cost"],
        demand_low=worker_cfg["demand_low"], batch_size=worker_cfg["batch_size"],
        init_inventory=worker_cfg.get("init_inventory"),
    )

    if not os.path.exists(ckpt_path):
        _train_one(
            hidden_layers=worker_cfg["hidden_layers"], n_cycles=worker_cfg["n_cycles"],
            sourcing_model=sm, sourcing_periods=worker_cfg["sourcing_periods"],
            epochs=worker_cfg["epochs"], parameters_lr=worker_cfg["parameters_lr"],
            init_inventory_lr=worker_cfg["init_inventory_lr"], seed=seed,
            checkpoint_path=ckpt_path, base_checkpoint=worker_cfg["base_checkpoint"],
            device=worker_cfg["device"], patience=worker_cfg["patience"],
            use_scheduler=worker_cfg["use_scheduler"], target_vf=worker_cfg.get("target_vf"),
            vf_gap_tol=worker_cfg.get("vf_gap_tol", VF_GAP_TOL),
            vf_patience=worker_cfg.get("vf_patience", VF_PATIENCE),
        )

    esm = _build_sourcing_model(
        worker_cfg["lt_s"], worker_cfg["n_cycles"], worker_cfg["shortage_cost"],
        demand_high=worker_cfg["demand_high"], expedited_cost=worker_cfg["expedited_cost"],
        demand_low=worker_cfg["demand_low"], batch_size=worker_cfg["batch_size"],
        init_inventory=worker_cfg.get("init_inventory"),
    )
    ctrl = CyclicDualNeuralController.load_checkpoint(ckpt_path, esm, device=worker_cfg["device"])
    costs = []
    with torch.no_grad():
        for es in range(EVAL_SEEDS):
            costs.append(ctrl.get_average_cost(esm, EVAL_PERIODS, seed=es))

    mu = torch.stack(costs).mean().item()
    std = torch.stack(costs).std().item()
    return {"seed": seed, "mean_cost": mu, "std_cost": std, "ckpt_path": ckpt_path}


def run_full(args):
    """Execute complete training run (scratch or transfer fine-tuning)."""
    os.makedirs(args.checkpoint_dir, exist_ok=True)

    # Determine learning rate
    lr = args.lr if args.lr is not None else (2e-4 if args.base_checkpoint else 1e-3)

    # Determine hidden layer architecture
    if args.hidden_layers:
        hl = [int(x) for x in args.hidden_layers.split(",")]
    else:
        hl = DEFAULT_HIDDEN_LAYERS

    sp = args.sourcing_periods or _sourcing_periods_for(args.lt_s)
    epochs = args.epochs or DEFAULT_EPOCHS
    best = os.path.join(args.checkpoint_dir, "best_model.pt")

    logger.info("Running training: lt_s=%d, N=%d, b=%.1f, lr=%.6f, layers=%s, epochs=%d, device=%s",
                args.lt_s, args.n_cycles, args.shortage_cost, lr, hl, epochs, args.device)

    if args.n_seeds <= 1:
        sm = _build_sourcing_model(
            args.lt_s, args.n_cycles, args.shortage_cost,
            demand_high=args.demand_high, expedited_cost=args.expedited_cost,
            demand_low=args.demand_low, batch_size=args.batch_size,
            init_inventory=args.init_inventory,
        )
        _train_one(
            hidden_layers=hl, n_cycles=args.n_cycles, sourcing_model=sm,
            sourcing_periods=sp, epochs=epochs, parameters_lr=lr,
            init_inventory_lr=args.init_inventory_lr, seed=args.seed,
            checkpoint_path=best, base_checkpoint=args.base_checkpoint,
            device=args.device, patience=args.patience, use_scheduler=args.use_scheduler,
            target_vf=args.vf, vf_gap_tol=args.vf_gap_tol, vf_patience=args.vf_patience,
        )
        print(f"\nModel checkpoint saved -> {best}")
        return

    # Multi-seed execution
    sd = os.path.join(args.checkpoint_dir, "seeded")
    os.makedirs(sd, exist_ok=True)
    worker_cfgs = [
        dict(
            seed=s, hidden_layers=hl, n_cycles=args.n_cycles, lt_s=args.lt_s,
            shortage_cost=args.shortage_cost, expedited_cost=args.expedited_cost,
            demand_low=args.demand_low, demand_high=args.demand_high,
            batch_size=args.batch_size, init_inventory=args.init_inventory,
            sourcing_periods=sp, epochs=epochs, parameters_lr=lr,
            init_inventory_lr=args.init_inventory_lr, base_checkpoint=args.base_checkpoint,
            device=args.device, target_vf=args.vf, vf_gap_tol=args.vf_gap_tol,
            vf_patience=args.vf_patience, patience=args.patience,
            use_scheduler=args.use_scheduler, ckpt_path=os.path.join(sd, f"model_seed{s}.pt"),
        )
        for s in range(args.n_seeds)
    ]

    results = []
    parallel = max(1, args.parallel_seeds)

    if parallel <= 1:
        for cfg in worker_cfgs:
            r = _seed_worker(cfg)
            results.append(r)
    else:
        ctx = mp.get_context("spawn")
        with concurrent.futures.ProcessPoolExecutor(max_workers=parallel, mp_context=ctx) as executor:
            future_to_seed = {executor.submit(_seed_worker, cfg): cfg["seed"] for cfg in worker_cfgs}
            for future in concurrent.futures.as_completed(future_to_seed):
                results.append(future.result())

    valid = [r for r in results if math.isfinite(r["mean_cost"])]
    if not valid:
        raise RuntimeError("All seeds diverged (NaN/Inf cost). Try lowering learning rate.")

    valid.sort(key=lambda r: r["mean_cost"])
    best_r = valid[0]
    shutil.copy(best_r["ckpt_path"], best)
    print(f"\nBest seed: {best_r['seed']} | Cost: {best_r['mean_cost']:.4f} (std {best_r['std_cost']:.4f}) -> {best}")


def run_infer(args):
    """Evaluate pre-trained model over 500 test seeds and compute certified GAP%."""
    best = os.path.join(args.checkpoint_dir, "best_model.pt")
    if not os.path.exists(best):
        raise FileNotFoundError(f"Checkpoint not found: {best!r}. Run training first.")

    sm = _build_sourcing_model(
        args.lt_s, args.n_cycles, args.shortage_cost,
        demand_high=args.demand_high, expedited_cost=args.expedited_cost,
        demand_low=args.demand_low, batch_size=args.batch_size,
        init_inventory=args.init_inventory,
    )
    ctrl = CyclicDualNeuralController.load_checkpoint(best, sm, device=args.device)

    costs = []
    with torch.no_grad():
        for seed in tqdm(range(EVAL_SEEDS), desc=f"Evaluating policy (500 seeds)"):
            costs.append(ctrl.get_average_cost(sm, EVAL_PERIODS, seed=seed))

    mu = torch.stack(costs).mean().item()
    std = torch.stack(costs).std().item()

    print(f"\n{'=' * 60}")
    print(f"Policy Evaluation (N={args.n_cycles}, l_s={args.lt_s}, b={args.shortage_cost}, D~U(0,{args.demand_high})):")
    print(f"  Simulated Mean Cost : {mu:.4f}")
    print(f"  Standard Deviation  : {std:.4f}")
    if args.vf is not None:
        gap = (mu - args.vf) / args.vf * 100.0
        print(f"  VF Value     : {args.vf:.4f}")
        print(f"  Optimality GAP      : {gap:.4f}%")
    print(f"{'=' * 60}\n")


def _parse_args():
    p = argparse.ArgumentParser(
        description="Neural Network Training and Transfer Learning for Cyclic Dual-Sourcing",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # Primary minimal CLI arguments
    p.add_argument("--mode", choices=["train", "infer", "scan", "full"], default="train",
                   help="Operation mode: 'train' (training/transfer), 'infer' (evaluation), or 'scan'")
    p.add_argument("--n_cycles", "-N", type=int, default=3, help="Replenishment cycle length N")
    p.add_argument("--lt_s", "-l", type=int, required=True, help="Regular order lead time l_s")
    p.add_argument("--shortage_cost", "-b", type=float, required=True, help="Unit shortage penalty b")
    p.add_argument("--demand_high", "-d", type=int, default=4, help="Upper bound of uniform demand U(0, D_max)")
    p.add_argument("--lr", type=float, default=None,
                   help="Learning rate (default: 1e-3 for scratch training, 2e-4 for transfer fine-tuning)")
    p.add_argument("--base_checkpoint", type=str, default=None,
                   help="Path to pre-trained base model for transfer learning")
    p.add_argument("--checkpoint_dir", type=str, default="models/default",
                   help="Directory for saving and loading model checkpoints and logs")
    p.add_argument("--device", type=str, default=DEFAULT_DEVICE, choices=["cpu", "cuda"],
                   help="Computation device ('cuda' or 'cpu')")
    p.add_argument("--vf", type=float, default=None,
                   help="Certified DP value-function baseline (for GAP%% reporting)")

    # Legacy/Extended options supported cleanly for backward compatibility
    p.add_argument("--parameters_lr", type=float, default=None, help=argparse.SUPPRESS)
    p.add_argument("--epochs", type=int, default=None, help=argparse.SUPPRESS)
    p.add_argument("--patience", type=int, default=DEFAULT_PATIENCE, help=argparse.SUPPRESS)
    p.add_argument("--hidden_layers", type=str, default=None, help=argparse.SUPPRESS)
    p.add_argument("--demand_low", type=int, default=DEMAND_LOW, help=argparse.SUPPRESS)
    p.add_argument("--expedited_cost", type=float, default=EXPEDITED_COST, help=argparse.SUPPRESS)
    p.add_argument("--batch_size", type=int, default=BATCH_SIZE, help=argparse.SUPPRESS)
    p.add_argument("--init_inventory", type=float, default=None, help=argparse.SUPPRESS)
    p.add_argument("--init_inventory_lr", type=float, default=INIT_INVENTORY_LR, help=argparse.SUPPRESS)
    p.add_argument("--sourcing_periods", type=int, default=None, help=argparse.SUPPRESS)
    p.add_argument("--seed", type=int, default=42, help=argparse.SUPPRESS)
    p.add_argument("--n_seeds", type=int, default=1, help=argparse.SUPPRESS)
    p.add_argument("--parallel_seeds", type=int, default=1, help=argparse.SUPPRESS)
    p.add_argument("--use_scheduler", action="store_true", default=False, help=argparse.SUPPRESS)
    p.add_argument("--vf_gap_tol", type=float, default=VF_GAP_TOL, help=argparse.SUPPRESS)
    p.add_argument("--vf_patience", type=int, default=VF_PATIENCE, help=argparse.SUPPRESS)

    args = p.parse_args()

    # Synchronize alias arguments
    if args.parameters_lr is not None and args.lr is None:
        args.lr = args.parameters_lr

    return args


def main():
    args = _parse_args()

    os.makedirs(args.checkpoint_dir, exist_ok=True)
    run_log_path = os.path.join(args.checkpoint_dir, "train.log")
    run_handler = logging.FileHandler(run_log_path, mode="w")
    run_handler.setLevel(logging.INFO)
    run_handler.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s"))
    logging.getLogger().addHandler(run_handler)

    if args.device == "cuda" and not torch.cuda.is_available():
        logger.warning("CUDA requested but not available. Falling back to CPU.")
        args.device = "cpu"

    if args.mode in ("train", "full"):
        run_full(args)
    elif args.mode == "infer":
        run_infer(args)
    elif args.mode == "scan":
        run_scan(args)


if __name__ == "__main__":
    main()
