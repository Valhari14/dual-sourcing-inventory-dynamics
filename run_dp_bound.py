"""
run_dp_bound.py

Runs the cycle-unrolled dynamic programming dual-sourcing controller (Problem II)
for a parameter set and saves the certified optimal value function (VF) to JSON.

Usage:
    python run_dp_bound.py \
        --cycle_length 2 \
        --regular_lead_time 2 \
        --backlog_cost 495 \
        --demand_max 4 \
        --output_path results/problem2_dp_bound.json
"""

import argparse
import json
import logging
import os
import pickle
import sys
import time
from datetime import datetime

# Ensure project root is on sys.path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# -----------------------------------------------------------------------------
# Dual-Sourcing Benchmark Constants (Problem II)
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
        description="Run cyclic dual-sourcing dynamic programming (Problem II)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--cycle_length", type=int, default=2, help="Replenishment cycle length N")
    parser.add_argument("--regular_lead_time", type=int, default=2, help="Regular lead time l_s")
    parser.add_argument("--backlog_cost", type=float, default=495.0, help="Unit shortage penalty b")
    parser.add_argument("--demand_max", type=int, default=4, help="Upper bound for uniform demand U(0, d_max)")
    parser.add_argument("--controller", choices=["parallel", "serial"], default="parallel", help="DP solver engine")
    parser.add_argument("--output_path", type=str, default="results/dp_bound/dp_result.json", help="Path to save results")
    parser.add_argument("--save_qf", action="store_true", help="Save full Q-factor policy dictionary")
    return parser.parse_args()


def main():
    args = parse_args()

    logger.info("=" * 60)
    logger.info("Problem II DP Bound Parameters:")
    logger.info("  Cycle Length (N)   : %d", args.cycle_length)
    logger.info("  Regular Lead Time  : %d", args.regular_lead_time)
    logger.info("  Backlog Penalty (b): %.1f", args.backlog_cost)
    logger.info("  Demand             : U(%d, %d)", DEMAND_MIN, args.demand_max)
    logger.info("  Controller Engine  : %s", args.controller)
    logger.info("=" * 60)

    from src.idinn.demand import UniformDemand
    from src.idinn.sourcing_model import DualSourcingModel

    if args.controller == "serial":
        from src.idinn.cyclic_dual_controller.dp_bound import DynamicProgrammingController
    else:
        from src.idinn.cyclic_dual_controller.dp_bound_parallel import DynamicProgrammingController

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

    controller = DynamicProgrammingController(cycle_length=args.cycle_length)

    logger.info("Fitting dynamic programming value function...")
    t0 = time.time()
    controller.fit(
        sourcing_model=model,
        max_iterations=MAX_ITERATIONS,
        tolerance=TOLERANCE,
        validation_freq=VALIDATION_FREQ,
        log_freq=LOG_FREQ,
        bound_slack=1,
    )
    fit_duration = time.time() - t0
    logger.info("DP fit completed in %.1fs (%.2fh)", fit_duration, fit_duration / 3600.0)

    # Evaluate simulated average cost under optimal policy
    avg_cost = controller.get_average_cost(
        sourcing_model=model,
        sourcing_periods=SOURCING_PERIODS,
        seed=SEED,
    )
    avg_cost_val = avg_cost.detach().item()
    logger.info("Simulated average cost per period: %.4f", avg_cost_val)

    results = {
        "timestamp": datetime.now().isoformat(),
        "controller": args.controller,
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
        "num_states": len(controller.qf) if controller.qf else None,
    }

    with open(args.output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    logger.info("Results saved to %s", args.output_path)

    if args.save_qf:
        qf_path = args.output_path.replace(".json", "_qf.pkl")
        with open(qf_path, "wb") as f:
            pickle.dump({"qf": controller.qf, "vf": controller.vf}, f)
        logger.info("Policy (qf) saved to %s", qf_path)


if __name__ == "__main__":
    main()