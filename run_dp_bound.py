"""
run_dp_bound.py

Runs the cycle-unrolled DP dual-sourcing controller (either the vanilla
serial implementation in dynamic_programming.py, or the parallel one in
dp_bound_parallel.py) for a single parameter combination and saves results
to a JSON file.

Use --controller to pick which implementation runs. Run the SAME
parameters through both --controller serial and --controller parallel to
validate the parallel version before trusting it for a full sweep.

Usage example (small validation combo, serial):
    python run_dp_bound.py \
        --controller serial \
        --cycle_length 1 \
        --backlog_cost 50 \
        --holding_cost 5 \
        --expedited_order_cost 10 \
        --regular_lead_time 2 \
        --expedited_lead_time 0 \
        --regular_order_cost 0 \
        --demand_min 0 \
        --demand_max 2 \
        --bound_slack 1 \
        --max_iterations 5000 \
        --tolerance 1e-6 \
        --sourcing_periods 200 \
        --seed 42 \
        --output_path results/dp_bound/validation/serial_quick.json \
        --save_qf

Same command with --controller parallel and a different --output_path
gives you the matching parallel run to diff against.
"""

import argparse
import json
import logging
import os
import pickle
import time
from datetime import datetime

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="Run DP dual-sourcing (dp_bound) experiment")

    parser.add_argument(
        "--controller", choices=["serial", "parallel"], default="parallel",
        help="serial -> dynamic_programming.DynamicProgrammingController (vanilla), "
             "parallel -> dp_bound_parallel.DynamicProgrammingController",
    )

    # Model parameters
    parser.add_argument("--cycle_length", type=int, default=2)
    
    parser.add_argument("--backlog_cost", type=float, default=495.0)
    parser.add_argument("--holding_cost", type=float, default=5.0)
    parser.add_argument("--expedited_order_cost", type=float, default=20.0)
    parser.add_argument("--regular_lead_time", type=int, default=2)
    parser.add_argument("--expedited_lead_time", type=int, default=0)
    parser.add_argument("--regular_order_cost", type=float, default=0.0)
    parser.add_argument("--demand_min", type=int, default=0)
    parser.add_argument("--demand_max", type=int, default=4)

    # DP parameters
    parser.add_argument("--max_iterations", type=int, default=1_000_000)
    parser.add_argument("--tolerance", type=float, default=1e-7)
    parser.add_argument("--validation_freq", type=int, default=100)
    parser.add_argument("--log_freq", type=int, default=100)
    parser.add_argument(
        "--bound_slack", type=int, default=1,
        help="1 = base signed IP box, 2 = widened box for truncation-sensitivity re-run "
             "(matching values between 1 and 2 certify the truncation is non-binding).",
    )

    # Evaluation parameters
    parser.add_argument("--sourcing_periods", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)

    # Output
    parser.add_argument("--output_path", type=str, default="results/dp_bound/dp_result.json")
    parser.add_argument(
        "--save_qf", action="store_true",
        help="Also pickle the full qf policy dict + vf alongside output_path "
             "(needed to diff serial vs parallel state-by-state, not just the vf scalar).",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    logger.info("=" * 60)
    logger.info("Experiment parameters:")
    for k, v in vars(args).items():
        logger.info(f"  {k}: {v}")
    logger.info("=" * 60)

    from idinn.demand import UniformDemand
    from idinn.sourcing_model import DualSourcingModel

    if args.controller == "serial":
        from idinn.dual_controller.dp_bound import DynamicProgrammingController
    else:
        from idinn.dual_controller.dp_bound_parallel import DynamicProgrammingController

    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)

    demand = UniformDemand(low=args.demand_min, high=args.demand_max)

    model = DualSourcingModel(
        demand_generator=demand,
        regular_lead_time=args.regular_lead_time,
        expedited_lead_time=args.expedited_lead_time,
        regular_order_cost=args.regular_order_cost,
        expedited_order_cost=args.expedited_order_cost,
        holding_cost=args.holding_cost,
        shortage_cost=args.backlog_cost,
        init_inventory=0,
        batch_size=1,
    )

    controller = DynamicProgrammingController(cycle_length=args.cycle_length)

    logger.info(f"Starting DP fit ({args.controller})...")
    t0 = time.time()

    controller.fit(
        sourcing_model=model,
        max_iterations=args.max_iterations,
        tolerance=args.tolerance,
        validation_freq=args.validation_freq,
        log_freq=args.log_freq,
        bound_slack=args.bound_slack,
    )

    fit_duration = time.time() - t0
    logger.info(f"DP fit completed in {fit_duration:.1f}s ({fit_duration/3600:.2f}h)")

    logger.info("Evaluating average cost...")
    avg_cost = controller.get_average_cost(
        sourcing_model=model,
        sourcing_periods=args.sourcing_periods,
        seed=args.seed,
    )
    avg_cost_val = avg_cost.detach().item()
    logger.info(f"Average cost per period: {avg_cost_val:.4f}")

    results = {
        "timestamp": datetime.now().isoformat(),
        "controller": args.controller,
        "parameters": vars(args),
        "vf_value": controller.vf,
        "average_cost": avg_cost_val,
        "fit_duration_seconds": fit_duration,
        "num_states": len(controller.qf) if controller.qf else None,
    }

    with open(args.output_path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"Results saved to {args.output_path}")

    if args.save_qf:
        qf_path = args.output_path.replace(".json", "_qf.pkl")
        with open(qf_path, "wb") as f:
            pickle.dump({"qf": controller.qf, "vf": controller.vf}, f)
        logger.info(f"Policy (qf) saved to {qf_path}")

    logger.info("Done.")


if __name__ == "__main__":
    main()