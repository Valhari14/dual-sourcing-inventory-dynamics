"""
run_dp_programming.py

Runs DynamicProgrammingParityController for unrestricted dual-sourcing (Problem I)
and saves benchmark value function results to a JSON file.

Usage:
    python run_dp_programming.py \
        --cycle_length 2 \
        --regular_lead_time 2 \
        --backlog_cost 495 \
        --demand_max 4 \
        --output_path results/problem1_result.json
"""

import argparse
import json
import logging
import os
import sys
import time
from datetime import datetime

# Ensure project root is on sys.path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# -----------------------------------------------------------------------------
# Dual-Sourcing Benchmark Constants (Problem I)
# -----------------------------------------------------------------------------
HOLDING_COST = 5.0
EXPEDITED_ORDER_COST = 20.0
REGULAR_ORDER_COST = 0.0
EXPEDITED_LEAD_TIME = 0
DEMAND_MIN = 0
TOLERANCE = 1e-7
MAX_ITERATIONS = 1_000_000
VALIDATION_FREQ = 100
LOG_FREQ = 100
SOURCING_PERIODS = 1000
SEED = 42

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run unrestricted dual-sourcing dynamic programming (Problem I)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--cycle_length", type=int, default=2, help="Replenishment cycle length")
    parser.add_argument("--regular_lead_time", type=int, default=2, help="Regular lead time l_s")
    parser.add_argument("--backlog_cost", type=float, default=495.0, help="Unit shortage penalty b")
    parser.add_argument("--demand_max", type=int, default=4, help="Upper bound for uniform demand U(0, d_max)")
    parser.add_argument("--output_path", type=str, default="results/problem1_result.json", help="Path to save results")
    return parser.parse_args()


def main():
    args = parse_args()

    logger.info("=" * 60)
    logger.info("Problem I DP Experiment Parameters:")
    logger.info("  Cycle Length       : %d", args.cycle_length)
    logger.info("  Regular Lead Time  : %d", args.regular_lead_time)
    logger.info("  Backlog Penalty (b): %.1f", args.backlog_cost)
    logger.info("  Demand             : U(%d, %d)", DEMAND_MIN, args.demand_max)
    logger.info("=" * 60)

    from src.idinn.demand import UniformDemand
    from src.idinn.sourcing_model import DualSourcingModel
    from src.idinn.dual_controller.dynamic_programming_parity import DynamicProgrammingParityController

    os.makedirs(os.path.dirname(os.path.abspath(args.output_path)), exist_ok=True)

    demand = UniformDemand(low=DEMAND_MIN, high=args.demand_max)
    model = DualSourcingModel(
        demand_generator=demand,
        regular_lead_time=args.regular_lead_time,
        expedited_lead_time=EXPEDITED_LEAD_TIME,
        regular_order_cost=REGULAR_ORDER_COST,
        expedited_order_cost=EXPEDITED_ORDER_COST,
        holding_cost=HOLDING_COST,
        shortage_cost=args.backlog_cost,
        init_inventory=0,
        batch_size=1,
    )

    controller = DynamicProgrammingParityController(cycle_length=args.cycle_length)

    logger.info("Starting DP fit...")
    t0 = time.time()
    controller.fit(
        sourcing_model=model,
        max_iterations=MAX_ITERATIONS,
        tolerance=TOLERANCE,
        validation_freq=VALIDATION_FREQ,
        log_freq=LOG_FREQ,
    )
    fit_duration = time.time() - t0
    logger.info("DP fit completed in %.1fs (%.2fh)", fit_duration, fit_duration / 3600.0)

    avg_cost = controller.get_average_cost(
        sourcing_model=model,
        sourcing_periods=SOURCING_PERIODS,
        seed=SEED,
    )
    avg_cost_val = avg_cost.detach().item()
    logger.info("Average cost per period: %.4f", avg_cost_val)

    results = {
        "timestamp": datetime.now().isoformat(),
        "parameters": {
            "cycle_length": args.cycle_length,
            "regular_lead_time": args.regular_lead_time,
            "backlog_cost": args.backlog_cost,
            "holding_cost": HOLDING_COST,
            "expedited_order_cost": EXPEDITED_ORDER_COST,
            "demand_min": DEMAND_MIN,
            "demand_max": args.demand_max,
        },
        "vf_value": controller.vf,
        "average_cost": avg_cost_val,
        "fit_duration_seconds": fit_duration,
    }

    with open(args.output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    logger.info("Results saved to %s", args.output_path)


if __name__ == "__main__":
    main()