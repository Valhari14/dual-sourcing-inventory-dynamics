import logging
import argparse
import torch
from tqdm import tqdm
import os

from src.idinn.cyclic_dual_controller.cyclic_dual_neural import CyclicDualNeuralController
from src.idinn.sourcing_model import DualSourcingModel
from src.idinn.demand import UniformDemand

# ---------------------------------------------------------------------
# logging
# ---------------------------------------------------------------------
logging.basicConfig(
    filename="src/idinn/finetuning/tune_cyclic_neural.log",
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger(__name__)

CHECKPOINT_PATH = "models/trained/best_model.pt"
SEEDED_DIR = "models/trained/seeded"
BEST_MODEL_PATH = "models/trained/best_model.pt"
N_SEEDS = 150
EVAL_PERIODS = 1000
EVAL_SEEDS = 500


def get_sourcing_model() -> DualSourcingModel:
    return DualSourcingModel(
        regular_lead_time=2,
        expedited_lead_time=0,
        regular_order_cost=0,
        expedited_order_cost=20,
        holding_cost=5,
        shortage_cost=95,
        init_inventory=6,
        demand_generator=UniformDemand(0, 8),
        batch_size=256,
    )


def train():
    if os.path.exists(CHECKPOINT_PATH):
        raise RuntimeError(
            f"Path : {CHECKPOINT_PATH} already exists! "
            "Please select a different name, or move the file"
        )

    sourcing_model = get_sourcing_model()
    controller = CyclicDualNeuralController(hidden_layers=[64, 32, 16, 8,4], n_cycles=3)
    controller.fit(
        sourcing_model=sourcing_model,
        sourcing_periods=100,
        epochs=5500,
        validation_sourcing_periods=1000,
        parameters_lr=3e-4,
        seed=42,
        checkpoint_path=CHECKPOINT_PATH,
    )
    print(f"Training complete. Checkpoint saved to {CHECKPOINT_PATH}")


def seed_train():
    os.makedirs(SEEDED_DIR, exist_ok=True)

    best_cost = float('inf')
    best_seed = None

    for i in range(100, N_SEEDS+1):
        seed = i - 1  # seeds 0..9
        checkpoint_path = os.path.join(SEEDED_DIR, f"model{i}.pt")

        if os.path.exists(checkpoint_path):
            logger.info(f"Seed {seed} — checkpoint already exists at {checkpoint_path}, skipping training.")
            print(f"[Seed {seed}] Checkpoint exists, skipping training.")
        else:
            print(f"\n[Seed {seed}] Training model {i}/{N_SEEDS}...")
            logger.info(f"Seed {seed} — starting training, saving to {checkpoint_path}")
            sourcing_model = get_sourcing_model()
            controller = CyclicDualNeuralController(hidden_layers=[64, 32, 16, 8,4], n_cycles=3)
            controller.fit(
                sourcing_model=sourcing_model,
                sourcing_periods=100,
                epochs=5500,
                validation_sourcing_periods=1000,
                parameters_lr=3e-4,
                seed=seed,
                checkpoint_path=checkpoint_path,
            )

        # evaluate
        print(f"[Seed {seed}] Evaluating...")
        sourcing_model = get_sourcing_model()
        controller = CyclicDualNeuralController.load_checkpoint(
            path=checkpoint_path,
            sourcing_model=sourcing_model,
        )

        costs = []
        with torch.no_grad():
            for eval_seed in range(EVAL_SEEDS):
                cost = controller.get_average_cost(
                    sourcing_model=sourcing_model,
                    sourcing_periods=EVAL_PERIODS,
                    seed=eval_seed,
                )
                costs.append(cost)

        mean_cost = torch.mean(torch.stack(costs)).item()
        std_cost = torch.std(torch.stack(costs)).item()

        logger.info(f"Seed {seed} — mean cost: {mean_cost:.4f}, std: {std_cost:.4f}")
        print(f"[Seed {seed}] Mean: {mean_cost:.4f}, Std: {std_cost:.4f}")

        if mean_cost < best_cost:
            best_cost = mean_cost
            best_seed = seed
            import shutil
            shutil.copy(checkpoint_path, BEST_MODEL_PATH)
            logger.info(f"New best model — seed {seed}, mean cost {mean_cost:.4f}, saved to {BEST_MODEL_PATH}")
            print(f"[Seed {seed}] New best! Saved to {BEST_MODEL_PATH}")

    print(f"\nSeed training complete.")
    print(f"Best seed: {best_seed}, Best mean cost: {best_cost:.4f}")
    print(f"Best model saved to {BEST_MODEL_PATH}")
    logger.info(f"Seed training complete. Best seed: {best_seed}, best cost: {best_cost:.4f}")


def infer():
    sourcing_model = get_sourcing_model()
    controller = CyclicDualNeuralController.load_checkpoint(
        path=CHECKPOINT_PATH,
        sourcing_model=sourcing_model,
    )

    costs = []
    with torch.no_grad():
        for seed in tqdm(range(500)):
            cost = controller.get_average_cost(
                sourcing_model=sourcing_model,
                sourcing_periods=1000,
                seed=seed,
            )
            costs.append(cost)
            logger.info(f"Inference Cost : {cost:.4f}")

    mean_cost = torch.mean(torch.stack(costs))
    std_cost = torch.std(torch.stack(costs))

    logger.info(f"Final mean cost: {mean_cost}")
    logger.info(f"Final std cost: {std_cost}")
    print("Final mean:", mean_cost.item())
    print("Final std:", std_cost.item())


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Cyclic Neural Controller")
    parser.add_argument("--train", action="store_true", help="Train a single model and save checkpoint")
    parser.add_argument("--seed_train", action="store_true", help="Train across multiple seeds, save best model")
    parser.add_argument("--infer", action="store_true", help="Load checkpoint and run inference")
    args = parser.parse_args()

    if args.train:
        train()
    elif args.seed_train:
        seed_train()
    elif args.infer:
        infer()
    else:
        parser.print_help()