"""
run_dp_experiment.py

Runs DynamicProgrammingParityController for a single parameter combination
and saves results to a JSON file.

Usage example (matches the SLURM script):
    python run_dp_experiment.py \
        --cycle_length 2 \
        --backlog_cost 495 \
        --holding_cost 5 \
        --expedited_order_cost 20 \
        --regular_lead_time 2 \
        --expedited_lead_time 0 \
        --regular_order_cost 0 \
        --demand_min 0 \
        --demand_max 4 \
        --max_iterations 1000000 \
        --tolerance 1e-7 \
        --sourcing_periods 1000 \
        --seed 42 \
        --output_path results/cycle2_model1_b495_lr2_U04.json
"""

import argparse
import json
import logging
import os
import time
from datetime import datetime

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="Run DP dual-sourcing experiment")

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

    # Evaluation parameters
    parser.add_argument("--sourcing_periods", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)

    # Output
    parser.add_argument("--output_path", type=str, default="results/dp_result.json")

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
    from idinn.dual_controller.dynamic_programming_parity import DynamicProgrammingParityController

    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)

    # Build demand generator
    demand = UniformDemand(
        low=args.demand_min,
        high=args.demand_max,
    )

    # Build sourcing model
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

    # Build and fit controller
    controller = DynamicProgrammingParityController(
        cycle_length=args.cycle_length,
    )

    logger.info("Starting DP fit...")
    t0 = time.time()

    controller.fit(
        sourcing_model=model,
        max_iterations=args.max_iterations,
        tolerance=args.tolerance,
        validation_freq=args.validation_freq,
        log_freq=args.log_freq,
    )

    fit_duration = time.time() - t0
    logger.info(f"DP fit completed in {fit_duration:.1f}s ({fit_duration/3600:.2f}h)")

    # Evaluate
    logger.info("Evaluating average cost...")
    avg_cost = controller.get_average_cost(
        sourcing_model=model,
        sourcing_periods=args.sourcing_periods,
        seed=args.seed,
    )
    #avg_cost_val = float(avg_cost)
    avg_cost_val = avg_cost.detach().item()
    logger.info(f"Average cost per period: {avg_cost_val:.4f}")

    # Save results
    results = {
        "timestamp": datetime.now().isoformat(),
        "parameters": vars(args),
        "vf_value": controller.vf,
        "average_cost": avg_cost_val,
        "fit_duration_seconds": fit_duration,
    }

    with open(args.output_path, "w") as f:
        json.dump(results, f, indent=2)

    logger.info(f"Results saved to {args.output_path}")
    logger.info("Done.")


if __name__ == "__main__":
    main()