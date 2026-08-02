import logging
import argparse
import torch
from tqdm import tqdm
import os

from src.idinn.cyclic_dual_controller.cyclic_dual_neural import CyclicDualNeuralController
from src.idinn.sourcing_model import DualSourcingModel
from src.idinn.demand import UniformDemand
import itertools

# ---------------------------------------------------------------------
# Sourcing model parameters — edit these to configure the finetuning run
# ---------------------------------------------------------------------
REGULAR_LEAD_TIME     = 3      # options: 2, 3, 4
EXPEDITED_LEAD_TIME   = 0
REGULAR_ORDER_COST    = 0
EXPEDITED_ORDER_COST  = 20     # e.g. 5, 10, 20
HOLDING_COST          = 5
SHORTAGE_COST         = 495   # e.g. 95, 495
INIT_INVENTORY        = 0
DEMAND_LOW            = 0
DEMAND_HIGH           = 4      # options: 4, 8
N_CYCLES              =3    # options: 2, 3  (controls output heads)

# ---------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------
PRETRAINED_PATH = "models/trained/finetuned_lt4_nc2_sc95_ec20_d0-4.pt"
FINETUNED_DIR   = "models/trained/finetuned"


# Auto-generated name encodes the key parameters
_model_name = (
    f"finetuned"
    f"_lt{REGULAR_LEAD_TIME}"
    f"_nc{N_CYCLES}"
    f"_sc{SHORTAGE_COST}"
    f"_ec{EXPEDITED_ORDER_COST}"
    f"_d{DEMAND_LOW}-{DEMAND_HIGH}.pt"
)
FINETUNED_PATH = os.path.join(FINETUNED_DIR, _model_name)

# ---------------------------------------------------------------------
# Training hyperparameters
# ---------------------------------------------------------------------
FINETUNE_EPOCHS               = 6000
FINETUNE_SOURCING_PERIODS     = 100
FINETUNE_VALIDATION_PERIODS   = 1000
FINETUNE_PARAMETERS_LR        = 4e-4   # lower LR to preserve pretrained features
FINETUNE_INIT_INVENTORY_LR    = 1e-1
FINETUNE_SEED                 = 7

EVAL_PERIODS = 1000
EVAL_SEEDS   = 500

# ---------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------
logging.basicConfig(
    filename="src/idinn/finetuning/finetuning.log",
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger(__name__)


def get_sourcing_model() -> DualSourcingModel:
    return DualSourcingModel(
        regular_lead_time=REGULAR_LEAD_TIME,
        expedited_lead_time=EXPEDITED_LEAD_TIME,
        regular_order_cost=REGULAR_ORDER_COST,
        expedited_order_cost=EXPEDITED_ORDER_COST,
        holding_cost=HOLDING_COST,
        shortage_cost=SHORTAGE_COST,
        init_inventory=INIT_INVENTORY,
        demand_generator=UniformDemand(DEMAND_LOW, DEMAND_HIGH),
        batch_size=256,
    )


def _output_layer_key(hidden_layers: list) -> str:
    """
    Return the state-dict key prefix for the output Linear layer.

    The Sequential is built as:
      [0] Linear(in, h0), [1] act,
      [2] Linear(h0, h1), [3] act, ...
      [2*n] Linear(h_{n-1}, out), [2*n+1] ReLU
    so the output layer index is 2 * len(hidden_layers).
    """
    return f"model.{2 * len(hidden_layers)}"


def _load_pretrained(sourcing_model: DualSourcingModel) -> CyclicDualNeuralController:
    """
    Load pretrained weights from PRETRAINED_PATH into a controller sized for
    sourcing_model and N_CYCLES.

    Dimension mismatches are handled layer by layer:
    - Input layer  (model.0): re-initialized if regular_lead_time changed
    - Output layer (model.{2n}): re-initialized if n_cycles changed
    All intermediate hidden layers are always loaded from the checkpoint.
    """
    if not os.path.exists(PRETRAINED_PATH):
        raise FileNotFoundError(
            f"Pretrained checkpoint not found at '{PRETRAINED_PATH}'. "
            "Run pre_training.py --train or --seed_train first."
        )

    checkpoint = torch.load(PRETRAINED_PATH, map_location='cpu')
    pretrained_state  = checkpoint['model_state_dict']
    pretrained_hidden = checkpoint['hidden_layers']
    pretrained_cycles = checkpoint.get('n_cycles', 2)

    controller = CyclicDualNeuralController(
        hidden_layers=pretrained_hidden,
        n_cycles=N_CYCLES,
    )
    controller.init_layers(
        regular_lead_time=sourcing_model.get_regular_lead_time(),
        expedited_lead_time=sourcing_model.get_expedited_lead_time(),
    )
    controller.sourcing_model = sourcing_model
    sourcing_model.init_inventory.data.fill_(checkpoint['init_inventory'])

    current_state = controller.state_dict()

    # Determine which boundary layers need re-initialisation
    skip_keys = set()

    pretrained_input_dim = pretrained_state['model.0.weight'].shape[1]
    current_input_dim    = current_state['model.0.weight'].shape[1]
    if pretrained_input_dim != current_input_dim:
        skip_keys.update({'model.0.weight', 'model.0.bias'})
        logger.info(
            f"Input dimension changed ({pretrained_input_dim} → {current_input_dim}): "
            "first layer re-initialized."
        )
        print(
            f"Input dimension changed ({pretrained_input_dim} → {current_input_dim}): "
            "first layer re-initialized, deeper layers loaded from checkpoint."
        )

    out_key = _output_layer_key(pretrained_hidden)
    pretrained_output_dim = pretrained_state[f'{out_key}.weight'].shape[0]
    current_output_dim    = current_state[f'{out_key}.weight'].shape[0]
    if pretrained_output_dim != current_output_dim:
        skip_keys.update({f'{out_key}.weight', f'{out_key}.bias'})
        logger.info(
            f"Output dimension changed ({pretrained_output_dim} → {current_output_dim}, "
            f"n_cycles {pretrained_cycles} → {N_CYCLES}): output layer re-initialized."
        )
        print(
            f"n_cycles changed ({pretrained_cycles} → {N_CYCLES}), "
            f"output heads ({pretrained_output_dim} → {current_output_dim}): "
            "output layer re-initialized, all hidden layers loaded from checkpoint."
        )

    if skip_keys:
        compatible = {k: v for k, v in pretrained_state.items() if k not in skip_keys}
        current_state.update(compatible)
        controller.load_state_dict(current_state)
    else:
        controller.load_state_dict(pretrained_state)
        logger.info("All dimensions match — loaded all pretrained weights.")
        print("All dimensions match — loaded all pretrained weights.")

    return controller


def finetune():
    sourcing_model = get_sourcing_model()

    print(f"Loading pretrained weights from {PRETRAINED_PATH} ...")
    controller = _load_pretrained(sourcing_model)

    os.makedirs(FINETUNED_DIR, exist_ok=True)
    print(f"Fine-tuning → will save best checkpoint to {FINETUNED_PATH}")
    logger.info(
        f"Starting fine-tuning: lt={REGULAR_LEAD_TIME}, nc={N_CYCLES}, "
        f"sc={SHORTAGE_COST}, ec={EXPEDITED_ORDER_COST}, "
        f"demand=({DEMAND_LOW},{DEMAND_HIGH})"
    )

    controller.fit(
        sourcing_model=sourcing_model,
        sourcing_periods=FINETUNE_SOURCING_PERIODS,
        epochs=FINETUNE_EPOCHS,
        validation_sourcing_periods=FINETUNE_VALIDATION_PERIODS,
        parameters_lr=FINETUNE_PARAMETERS_LR,
        init_inventory_lr=FINETUNE_INIT_INVENTORY_LR,
        seed=FINETUNE_SEED,
        checkpoint_path=FINETUNED_PATH,
    )

    print(f"Fine-tuning complete. Checkpoint saved to {FINETUNED_PATH}")
    logger.info(f"Fine-tuning complete. Checkpoint saved to {FINETUNED_PATH}")


def infer():
    if not os.path.exists(FINETUNED_PATH):
        raise FileNotFoundError(
            f"Fine-tuned checkpoint not found at '{FINETUNED_PATH}'. "
            "Run with --finetune first."
        )

    logger.info(f"Inferring from finetuned model - {FINETUNED_PATH}")
    sourcing_model = get_sourcing_model()
    controller = CyclicDualNeuralController.load_checkpoint(
        path=FINETUNED_PATH,
        sourcing_model=sourcing_model,
    )

    costs = []
    with torch.no_grad():
        for seed in tqdm(range(EVAL_SEEDS)):
            cost = controller.get_average_cost(
                sourcing_model=sourcing_model,
                sourcing_periods=EVAL_PERIODS,
                seed=seed,
            )
            costs.append(cost)
            logger.info(f"Inference seed {seed} cost: {cost:.4f}")

    mean_cost = torch.mean(torch.stack(costs))
    std_cost  = torch.std(torch.stack(costs))

    logger.info(f"Final mean cost: {mean_cost:.4f}, std: {std_cost:.4f}")
    print(f"Final mean: {mean_cost.item():.4f}")
    print(f"Final std:  {std_cost.item():.4f}")

def run_all() -> None:
    global REGULAR_LEAD_TIME, SHORTAGE_COST, DEMAND_HIGH, N_CYCLES, FINETUNED_PATH, _model_name

    grid = {
        "REGULAR_LEAD_TIME": [2, 3, 4],
        "SHORTAGE_COST":     [95, 495],
        "DEMAND_HIGH":       [4, 8],
        "N_CYCLES":          [3, 4],
    }

    combos = list(itertools.product(*grid.values()))
    total  = len(combos)
    print(f"\nrun_all: {total} combinations\n{'='*60}")

    for idx, (lt, sc, dh, nc) in enumerate(combos, 1):
        REGULAR_LEAD_TIME, SHORTAGE_COST, DEMAND_HIGH, N_CYCLES = lt, sc, dh, nc

        _model_name    = f"finetuned_lt{lt}_nc{nc}_sc{sc}_ec{EXPEDITED_ORDER_COST}_d{DEMAND_LOW}-{dh}.pt"
        FINETUNED_PATH = os.path.join(FINETUNED_DIR, _model_name)

        print(f"\n[{idx}/{total}] lt={lt}, nc={nc}, sc={sc}, dh={dh}")

        try:
            finetune()
        except Exception as e:
            print(f"  ERROR during finetune: {e} — skipping."); continue

        try:
            infer()
        except Exception as e:
            print(f"  ERROR during infer: {e} — skipping.")

    print(f"\n{'='*60}\nrun_all complete.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Fine-tune CyclicDualNeuralController on a new cost/demand/cycle configuration"
    )
    parser.add_argument(
        "--finetune",
        action="store_true",
        help=f"Load {PRETRAINED_PATH} and fine-tune on the configured sourcing model",
    )
    parser.add_argument(
        "--infer",
        action="store_true",
        help=f"Load the finetuned checkpoint and run inference",
    )
    parser.add_argument("--run_all", action="store_true", help="Run finetune+infer for all parameter combinations")
    args = parser.parse_args()

    if args.finetune:
        finetune()
    elif args.infer:
        infer()
    elif args.run_all:
        run_all()
    else:
        parser.print_help()
