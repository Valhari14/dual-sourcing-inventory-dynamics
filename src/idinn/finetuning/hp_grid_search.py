"""
hp_grid_search.py  --  Hyperparameter grid search for CyclicDualNeuralController
Three modes: scan (6-combo grid), full (single combo + seed_train), infer (GAP%).
Device-agnostic: --device cpu | cuda.  Paper-correct: RMSprop, no scheduler, no grad-clip.

Usage (run from repo root):
  # Step 0a: CPU smoke test (2 epochs)
  python src/idinn/finetuning/hp_grid_search.py --mode scan --lt_s 2 --n_cycles 2 --shortage_cost 495 --epochs 2 --checkpoint_dir models/smoke_cpu --device cpu

  # Step 0b: GPU smoke test (same config, --device cuda)
  python src/idinn/finetuning/hp_grid_search.py --mode scan --lt_s 2 --n_cycles 2 --shortage_cost 495 --epochs 2 --checkpoint_dir models/smoke_gpu --device cuda

  # Step 1: Scan row (2,1) on GPU - 800 epochs
  python src/idinn/finetuning/hp_grid_search.py --mode scan --lt_s 2 --n_cycles 2 --shortage_cost 495 --epochs 800 --checkpoint_dir models/hp_grid/scan_row21 --device cuda

  # Step 2: Full run row (2,1) after picking winner from scan (example: lr=3e-3, paper layers)
  python src/idinn/finetuning/hp_grid_search.py --mode full --lt_s 2 --n_cycles 2 --shortage_cost 495 --parameters_lr 3e-3 --hidden_layers "128,64,32,16,8,4,2" --epochs 5500 --n_seeds 150 --checkpoint_dir models/hp_grid/full_row21 --device cuda

  # Step 3: Infer GAP% for row (2,1)
  python src/idinn/finetuning/hp_grid_search.py --mode infer --lt_s 2 --n_cycles 2 --shortage_cost 495 --vf 68.0055 --checkpoint_dir models/hp_grid/full_row21

  # Row (2,6): lt_s=3, n_cycles=2, b=95 -- uses sourcing_periods=150 automatically
  python src/idinn/finetuning/hp_grid_search.py --mode scan --lt_s 3 --n_cycles 2 --shortage_cost 95 --epochs 800 --checkpoint_dir models/hp_grid/scan_row26 --device cuda

  # Row (3,2): lt_s=2, n_cycles=3, b=95
  python src/idinn/finetuning/hp_grid_search.py --mode scan --lt_s 2 --n_cycles 3 --shortage_cost 95 --epochs 800 --checkpoint_dir models/hp_grid/scan_row32 --device cuda
"""

import argparse
import logging
import os
import shutil
import time
from typing import List, Optional

import torch
from tqdm import tqdm

from src.idinn.cyclic_dual_controller.cyclic_dual_neural import CyclicDualNeuralController
from src.idinn.sourcing_model import DualSourcingModel
from src.idinn.demand import UniformDemand

_LOG_FILE = "src/idinn/finetuning/hp_grid_search.log"
logging.basicConfig(filename=_LOG_FILE, level=logging.INFO,
                    format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

GRID_LRS: List[float] = [3e-4, 1e-3, 3e-3]
GRID_LAYERS: List[List[int]] = [
    [64, 32, 16, 8, 4],
    [128, 64, 32, 16, 8, 4, 2],
]

OPTIMIZER_TYPE: str = "rmsprop"
USE_SCHEDULER: bool  = False
USE_GRAD_CLIP: bool  = False

EVAL_PERIODS: int = 1000
EVAL_SEEDS: int   = 500


def _sourcing_periods_for(lt_s: int) -> int:
    """Paper EC.5.1: T=100/150/200 for lt_s=2/3/4."""
    if lt_s <= 2:   return 100
    elif lt_s == 3: return 150
    elif lt_s == 4: return 200
    else:           return 200 + (lt_s - 4) * 50


def _build_sourcing_model(lt_s, n_cycles, shortage_cost, expedited_cost,
                          demand_low, demand_high, batch_size):
    return DualSourcingModel(
        regular_lead_time=lt_s, expedited_lead_time=0,
        regular_order_cost=0, expedited_order_cost=expedited_cost,
        holding_cost=5, shortage_cost=shortage_cost,
        init_inventory=0,
        demand_generator=UniformDemand(demand_low, demand_high),
        batch_size=batch_size,
    )


def _load_pretrained(base_checkpoint, hidden_layers, n_cycles, sourcing_model):
    if not os.path.exists(base_checkpoint):
        raise FileNotFoundError(f"Base checkpoint not found: {base_checkpoint!r}. "
                                "Train base first (--mode full without --base_checkpoint).")
    ckpt = torch.load(base_checkpoint, map_location="cpu")
    ps   = ckpt["model_state_dict"]
    ph   = ckpt["hidden_layers"]
    pc   = ckpt.get("n_cycles", 2)

    ctrl = CyclicDualNeuralController(hidden_layers=hidden_layers, n_cycles=n_cycles)
    ctrl.init_layers(regular_lead_time=sourcing_model.get_regular_lead_time(),
                     expedited_lead_time=sourcing_model.get_expedited_lead_time())
    ctrl.sourcing_model = sourcing_model
    sourcing_model.init_inventory.data.fill_(ckpt["init_inventory"])

    cs   = ctrl.state_dict()
    skip: set = set()

    pi = ps["model.0.weight"].shape[1]
    ci = cs["model.0.weight"].shape[1]
    if pi != ci:
        skip.update({"model.0.weight", "model.0.bias"})
        msg = f"Input dim {pi}->{ci}: re-init first layer."
        logger.info(msg); print(msg)

    ok = f"model.{2 * len(ph)}"
    po = ps[f"{ok}.weight"].shape[0]
    co = cs[f"{ok}.weight"].shape[0]
    if po != co:
        skip.update({f"{ok}.weight", f"{ok}.bias"})
        msg = f"Output dim {po}->{co} (n_cycles {pc}->{n_cycles}): re-init output layer."
        logger.info(msg); print(msg)

    if skip:
        compat = {k: v for k, v in ps.items() if k not in skip}
        cs.update(compat)
        ctrl.load_state_dict(cs)
    else:
        ctrl.load_state_dict(ps)
        print("All dims match -- loaded all pretrained weights.")
    return ctrl


def _train_one(*, hidden_layers, n_cycles, sourcing_model, sourcing_periods,
               epochs, parameters_lr, init_inventory_lr, seed, checkpoint_path,
               base_checkpoint, device):
    if base_checkpoint:
        ctrl = _load_pretrained(base_checkpoint, hidden_layers, n_cycles, sourcing_model)
    else:
        ctrl = CyclicDualNeuralController(hidden_layers=hidden_layers, n_cycles=n_cycles)

    ctrl.fit(
        sourcing_model=sourcing_model,
        sourcing_periods=sourcing_periods,
        epochs=epochs,
        validation_sourcing_periods=1000,
        validation_freq=50,
        log_freq=10,
        init_inventory_lr=init_inventory_lr,
        parameters_lr=parameters_lr,
        seed=seed,
        checkpoint_path=checkpoint_path,
        optimizer_type=OPTIMIZER_TYPE,
        use_scheduler=USE_SCHEDULER,
        use_grad_clip=USE_GRAD_CLIP,
        device=device,
    )


def run_scan(args):
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    sp = args.sourcing_periods or _sourcing_periods_for(args.lt_s)
    print(f"\n{'='*64}\nSCAN  lt_s={args.lt_s} n_cycles={args.n_cycles} "
          f"b={args.shortage_cost} demand=U({args.demand_low},{args.demand_high})\n"
          f"optimizer={OPTIMIZER_TYPE} scheduler={USE_SCHEDULER} "
          f"grad_clip={USE_GRAD_CLIP} batch={args.batch_size}\n"
          f"T={sp} epochs={args.epochs} device={args.device}\n{'='*64}")
    logger.info("SCAN: lt_s=%d n_cycles=%d b=%d epochs=%d sp=%d device=%s",
                args.lt_s, args.n_cycles, args.shortage_cost, args.epochs, sp, args.device)

    results = []
    total = len(GRID_LRS) * len(GRID_LAYERS)
    c = 0
    for lr in GRID_LRS:
        for layers in GRID_LAYERS:
            c += 1
            ls   = ",".join(str(x) for x in layers)
            ckpt = os.path.join(args.checkpoint_dir,
                                f"scan_lr{lr}_layers{ls.replace(',','_')}.pt")
            print(f"\n[{c}/{total}] lr={lr} layers=[{ls}]")
            t0 = time.time()
            sm = _build_sourcing_model(args.lt_s, args.n_cycles, args.shortage_cost,
                                       args.expedited_cost, args.demand_low,
                                       args.demand_high, args.batch_size)
            _train_one(hidden_layers=layers, n_cycles=args.n_cycles,
                       sourcing_model=sm, sourcing_periods=sp,
                       epochs=args.epochs, parameters_lr=lr,
                       init_inventory_lr=args.init_inventory_lr,
                       seed=args.seed, checkpoint_path=ckpt,
                       base_checkpoint=args.base_checkpoint, device=args.device)
            elapsed = time.time() - t0
            esm = _build_sourcing_model(args.lt_s, args.n_cycles, args.shortage_cost,
                                        args.expedited_cost, args.demand_low,
                                        args.demand_high, args.batch_size)
            ec = CyclicDualNeuralController.load_checkpoint(ckpt, esm, device=args.device)
            with torch.no_grad():
                ev = [ec.get_average_cost(esm, 1000, seed=s) for s in range(20)]
            mu  = torch.stack(ev).mean().item()
            std = torch.stack(ev).std().item()
            results.append((mu, lr, layers, std, ckpt))
            print(f"  {elapsed/60:.1f}min | val_mean={mu:.4f} std={std:.4f}")
            logger.info("Combo %d: lr=%s layers=%s mu=%.4f std=%.4f t=%.0fs",
                        c, lr, layers, mu, std, elapsed)

    results.sort(key=lambda x: x[0])
    print(f"\n{'='*64}\nSCAN SUMMARY (lower val_mean = better)\n{'='*64}")
    print(f"{'Rk':<4} {'LR':<9} {'Layers':<30} {'ValMean':<11} ValStd")
    print("-"*60)
    for i, (mu, lr, layers, std, _) in enumerate(results, 1):
        tag = " <-- WINNER" if i == 1 else ""
        print(f"{i:<4} {lr:<9} {str(layers):<30} {mu:<11.4f} {std:.4f}{tag}")
    print(f"{'='*64}")

    bmu, blr, bl, bstd, bckpt = results[0]
    dest = os.path.join(args.checkpoint_dir, "best_scan_model.pt")
    shutil.copy(bckpt, dest)
    la = ",".join(str(x) for x in bl)
    print(f"\nWinner -> {dest}\n"
          f"  lr={blr}  layers={bl}\n"
          f"  val_mean={bmu:.4f} val_std={bstd:.4f}\n\n"
          f'Next: --mode full --parameters_lr {blr} --hidden_layers "{la}"')
    logger.info("Winner: lr=%s layers=%s val_mean=%.4f", blr, bl, bmu)


# ---------------------------------------------------------------------------
# Parallel seed worker  (TOP-LEVEL — must be picklable for multiprocessing)
# ---------------------------------------------------------------------------

def _seed_worker(worker_cfg: dict) -> dict:
    """
    Train ONE seed and evaluate it.  Runs inside a subprocess spawned by
    ProcessPoolExecutor so that multiple seeds execute concurrently on the
    same GPU.  Returns dict: {seed, mean_cost, std_cost, ckpt_path}.
    """
    import sys, os
    sys.path.insert(0, os.getcwd())          # ensure repo root is on path

    import torch
    import logging
    logging.disable(logging.CRITICAL)        # suppress log noise from workers

    seed          = worker_cfg["seed"]
    hl            = worker_cfg["hidden_layers"]
    n_cycles      = worker_cfg["n_cycles"]
    lt_s          = worker_cfg["lt_s"]
    shortage_cost = worker_cfg["shortage_cost"]
    expedited_cost= worker_cfg["expedited_cost"]
    demand_low    = worker_cfg["demand_low"]
    demand_high   = worker_cfg["demand_high"]
    batch_size    = worker_cfg["batch_size"]
    sp            = worker_cfg["sourcing_periods"]
    epochs        = worker_cfg["epochs"]
    parameters_lr = worker_cfg["parameters_lr"]
    init_inv_lr   = worker_cfg["init_inventory_lr"]
    base_ckpt     = worker_cfg["base_checkpoint"]
    device        = worker_cfg["device"]
    ckpt_path     = worker_cfg["ckpt_path"]

    sm = _build_sourcing_model(lt_s, n_cycles, shortage_cost, expedited_cost,
                               demand_low, demand_high, batch_size)

    if not os.path.exists(ckpt_path):
        _train_one(
            hidden_layers=hl, n_cycles=n_cycles,
            sourcing_model=sm, sourcing_periods=sp,
            epochs=epochs, parameters_lr=parameters_lr,
            init_inventory_lr=init_inv_lr,
            seed=seed, checkpoint_path=ckpt_path,
            base_checkpoint=base_ckpt, device=device,
        )

    # ---- evaluate -------------------------------------------------------
    from src.idinn.cyclic_dual_controller.cyclic_dual_neural import CyclicDualNeuralController
    esm  = _build_sourcing_model(lt_s, n_cycles, shortage_cost, expedited_cost,
                                 demand_low, demand_high, batch_size)
    ctrl = CyclicDualNeuralController.load_checkpoint(ckpt_path, esm, device=device)
    costs = []
    with torch.no_grad():
        for es in range(EVAL_SEEDS):
            costs.append(ctrl.get_average_cost(esm, EVAL_PERIODS, seed=es))
    mu  = torch.stack(costs).mean().item()
    std = torch.stack(costs).std().item()
    return {"seed": seed, "mean_cost": mu, "std_cost": std, "ckpt_path": ckpt_path}


def run_full(args):
    if args.parameters_lr is None: raise ValueError("--parameters_lr required for full")
    if args.hidden_layers  is None: raise ValueError("--hidden_layers required for full")
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    hl   = [int(x) for x in args.hidden_layers.split(",")]
    sp   = args.sourcing_periods or _sourcing_periods_for(args.lt_s)
    best = os.path.join(args.checkpoint_dir, "best_model.pt")
    parallel = max(1, args.parallel_seeds)

    print(
        f"\n{'='*64}\n"
        f"FULL  lt_s={args.lt_s} n_cycles={args.n_cycles} "
        f"b={args.shortage_cost} lr={args.parameters_lr} layers={hl}\n"
        f"T={sp} epochs={args.epochs} n_seeds={args.n_seeds} "
        f"parallel_seeds={parallel} device={args.device}\n"
        f"{'='*64}"
    )
    logger.info("FULL: lt_s=%d nc=%d b=%d lr=%s hl=%s ns=%d par=%d dev=%s",
                args.lt_s, args.n_cycles, args.shortage_cost, args.parameters_lr,
                hl, args.n_seeds, parallel, args.device)

    if args.n_seeds <= 1:
        # ---- single seed (original behaviour) ---------------------------
        sm = _build_sourcing_model(args.lt_s, args.n_cycles, args.shortage_cost,
                                   args.expedited_cost, args.demand_low,
                                   args.demand_high, args.batch_size)
        _train_one(hidden_layers=hl, n_cycles=args.n_cycles,
                   sourcing_model=sm, sourcing_periods=sp,
                   epochs=args.epochs, parameters_lr=args.parameters_lr,
                   init_inventory_lr=args.init_inventory_lr,
                   seed=args.seed, checkpoint_path=best,
                   base_checkpoint=args.base_checkpoint, device=args.device)
        print(f"Checkpoint -> {best}")
        return

    # ---- multi-seed: sequential or parallel -----------------------------
    sd = os.path.join(args.checkpoint_dir, "seeded")
    os.makedirs(sd, exist_ok=True)

    # Build list of per-seed configs
    worker_cfgs = [
        dict(
            seed=s,
            hidden_layers=hl,
            n_cycles=args.n_cycles,
            lt_s=args.lt_s,
            shortage_cost=args.shortage_cost,
            expedited_cost=args.expedited_cost,
            demand_low=args.demand_low,
            demand_high=args.demand_high,
            batch_size=args.batch_size,
            sourcing_periods=sp,
            epochs=args.epochs,
            parameters_lr=args.parameters_lr,
            init_inventory_lr=args.init_inventory_lr,
            base_checkpoint=args.base_checkpoint,
            device=args.device,
            ckpt_path=os.path.join(sd, f"model_seed{s}.pt"),
        )
        for s in range(args.n_seeds)
    ]

    results = []   # list of {seed, mean_cost, std_cost, ckpt_path}

    if parallel <= 1:
        # ---- sequential (fallback / CPU) --------------------------------
        for cfg in worker_cfgs:
            s = cfg["seed"]
            ckpt = cfg["ckpt_path"]
            print(f"\n[Seed {s}/{args.n_seeds-1}] Training..." if not os.path.exists(ckpt)
                  else f"[Seed {s}] Checkpoint exists -- skipping training.")
            r = _seed_worker(cfg)
            print(f"[Seed {s}] mean={r['mean_cost']:.4f}  std={r['std_cost']:.4f}")
            logger.info("Seed %d: mean=%.4f std=%.4f", s, r["mean_cost"], r["std_cost"])
            results.append(r)
    else:
        # ---- parallel via ProcessPoolExecutor (spawn context for CUDA) --
        import concurrent.futures
        import multiprocessing as _mp
        ctx = _mp.get_context("spawn")
        print(f"Launching {args.n_seeds} seeds across {parallel} parallel workers...")
        print("(Each worker trains one seed independently on the same GPU.)\n")

        with concurrent.futures.ProcessPoolExecutor(
            max_workers=parallel,
            mp_context=ctx,
        ) as executor:
            # Submit all seeds; executor queues them, running `parallel` at a time
            future_to_seed = {
                executor.submit(_seed_worker, cfg): cfg["seed"]
                for cfg in worker_cfgs
            }
            done = 0
            for future in concurrent.futures.as_completed(future_to_seed):
                s = future_to_seed[future]
                try:
                    r = future.result()
                    done += 1
                    print(f"[{done}/{args.n_seeds}] Seed {s} done: "
                          f"mean={r['mean_cost']:.4f}  std={r['std_cost']:.4f}")
                    logger.info("Seed %d: mean=%.4f std=%.4f", s, r["mean_cost"], r["std_cost"])
                    results.append(r)
                except Exception as exc:
                    print(f"[Seed {s}] FAILED: {exc}")
                    logger.error("Seed %d failed: %s", s, exc)

    # ---- pick global winner -----------------------------------------
    if not results:
        raise RuntimeError("All seeds failed — check logs.")

    results.sort(key=lambda r: r["mean_cost"])
    best_r = results[0]
    shutil.copy(best_r["ckpt_path"], best)

    print(f"\n{'='*64}")
    print(f"seed_train complete.")
    print(f"  Best seed  : {best_r['seed']}")
    print(f"  Mean cost  : {best_r['mean_cost']:.4f}")
    print(f"  Std        : {best_r['std_cost']:.4f}")
    print(f"  Checkpoint : {best}")
    print(f"{'='*64}\n")
    logger.info("seed_train done: best_seed=%d cost=%.4f -> %s",
                best_r["seed"], best_r["mean_cost"], best)



def run_infer(args):
    best = os.path.join(args.checkpoint_dir, "best_model.pt")
    if not os.path.exists(best):
        raise FileNotFoundError(f"No best_model.pt in {args.checkpoint_dir!r}. Run --mode full first.")
    print(f"\n{'='*64}\nINFER lt_s={args.lt_s} n_cycles={args.n_cycles} "
          f"b={args.shortage_cost} demand=U({args.demand_low},{args.demand_high})\n"
          f"checkpoint: {best}\n{'='*64}")
    sm   = _build_sourcing_model(args.lt_s, args.n_cycles, args.shortage_cost,
                                  args.expedited_cost, args.demand_low,
                                  args.demand_high, args.batch_size)
    ctrl = CyclicDualNeuralController.load_checkpoint(best, sm, device=args.device)
    cs   = []
    with torch.no_grad():
        for seed in tqdm(range(EVAL_SEEDS), desc="Evaluating"):
            cs.append(ctrl.get_average_cost(sm, EVAL_PERIODS, seed=seed))
    mu  = torch.stack(cs).mean().item()
    std = torch.stack(cs).std().item()
    print(f"\n{'='*64}")
    print(f"  NN mean cost : {mu:.4f}")
    print(f"  NN std       : {std:.4f}")
    if args.vf is not None:
        gap = (mu - args.vf) / args.vf * 100
        print(f"  VF baseline  : {args.vf:.4f}")
        print(f"  GAP%         : {gap:.4f}%")
        logger.info("Infer: mean=%.4f std=%.4f VF=%.4f GAP=%.4f%%", mu, std, args.vf, gap)
    else:
        logger.info("Infer: mean=%.4f std=%.4f", mu, std)
    print(f"{'='*64}\n")


def _parse_args():
    p = argparse.ArgumentParser(description="HP grid search for CyclicDualNeuralController",
                                formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--mode",           required=True, choices=["scan","full","infer"])
    p.add_argument("--lt_s",           type=int,   required=True)
    p.add_argument("--n_cycles",       type=int,   required=True)
    p.add_argument("--shortage_cost",  type=int,   required=True)
    p.add_argument("--expedited_cost", type=int,   default=20)
    p.add_argument("--demand_low",     type=int,   default=0)
    p.add_argument("--demand_high",    type=int,   default=4)
    p.add_argument("--sourcing_periods", type=int, default=None,
                   help="Auto-set from lt_s if omitted (100/150/200 for 2/3/4)")
    p.add_argument("--epochs",         type=int,   default=800)
    p.add_argument("--init_inventory_lr", type=float, default=1e-1)
    p.add_argument("--batch_size",     type=int,   default=512)
    p.add_argument("--seed",           type=int,   default=42)
    p.add_argument("--n_seeds",        type=int,   default=1,
                   help="Number of independent seeds to try; keep the best.")
    p.add_argument("--parallel_seeds", type=int,   default=1,
                   help=(
                       "How many seeds to train concurrently using separate processes "
                       "(all sharing the same GPU). Each process trains one seed. "
                       "Recommended: 4-8 for H100. 1 = sequential (default)."
                   ))
    p.add_argument("--parameters_lr",  type=float, default=None)
    p.add_argument("--hidden_layers",  type=str,   default=None,
                   help='e.g. "128,64,32,16,8,4,2"')
    p.add_argument("--base_checkpoint", type=str,  default=None)
    p.add_argument("--checkpoint_dir",  type=str,  default="models/hp_grid/default")
    p.add_argument("--vf",             type=float, default=None)
    p.add_argument("--device",         type=str,   default="cpu", choices=["cpu","cuda"],
                   help="HP logic is identical on cpu and cuda")
    return p.parse_args()


def main():
    args = _parse_args()
    if args.device == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("--device cuda but CUDA not available. Use --device cpu.")
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("Device: CPU")
    if args.mode == "scan":   run_scan(args)
    elif args.mode == "full": run_full(args)
    elif args.mode == "infer": run_infer(args)

if __name__ == "__main__":
    main()
