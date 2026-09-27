from typing import List, Optional, Tuple, Union
import logging 
from datetime import datetime 

import torch
import numpy as np
from tqdm import tqdm

from .base import BaseNeuralController
from ..sourcing_model import DualSourcingModel

# Get root logger
logger = logging.getLogger()

class CyclicDualNeuralController(torch.nn.Module, BaseNeuralController):
    """
    Implements a multi-period neural network architecture for cyclic dual-sourcing.
    The network receives the pipeline state and determines order quantities for the entire cycle.
    """

    def __init__(
        self, 
        hidden_layers: List[int] = [64, 32, 16, 8, 4],
        activation: torch.nn.Module = torch.nn.CELU(alpha=1.0),
        n_cycles: int = 2,
    ) -> None:
        """
        Parameters
        ----------
        hidden_layers: Architecture of hidden layers. hidden_layers[n] represents neurons in layer n.
        activation: Activation function between hidden layers.
        n_cycles: Number of periods in one replenishment cycle (output heads = n_cycles + 1).
        """
        super().__init__()

        self.hidden_layers = hidden_layers
        self.activation = activation 
        self.n_cycles = n_cycles 

        self.model = None

        assert self.n_cycles > 1, "Periods in a cycle should be > 1"

    def init_layers(self, regular_lead_time: int, expedited_lead_time: int) -> None:
        """
        Build NN architecture
        """

        input_length = regular_lead_time+expedited_lead_time+1

        architecture = [
            torch.nn.Linear(input_length, self.hidden_layers[0]),
            self.activation,
        ]
        for i in range(len(self.hidden_layers)):
            if i < len(self.hidden_layers) - 1:
                architecture += [
                    torch.nn.Linear(self.hidden_layers[i], self.hidden_layers[i + 1]),
                    self.activation,
                ]
        architecture += [
            torch.nn.Linear(self.hidden_layers[-1], self.n_cycles + 1),
            torch.nn.ReLU(),
        ]

        self.model = torch.nn.Sequential(*architecture)
        
        logger.info(
            f"Initialized neural network layers with regular_lead_time={regular_lead_time}, "
            f"expedited_lead_time={expedited_lead_time}, "
            f"Periods in a Cycle : {self.n_cycles}"
        )


    def prepare_inputs(
        self,
        current_inventory: torch.Tensor,
        past_regular_orders: torch.Tensor,
        past_expedited_orders: torch.Tensor,
        sourcing_model: DualSourcingModel,
    ) -> torch.Tensor:

        regular_lead_time = sourcing_model.get_regular_lead_time()
        expedited_lead_time = sourcing_model.get_expedited_lead_time()

        current_inventory = self._check_current_inventory(current_inventory)
        past_regular_orders = self._check_past_orders(
            past_regular_orders, regular_lead_time
        )
        past_expedited_orders = self._check_past_orders(
            past_expedited_orders, expedited_lead_time
        )

        if expedited_lead_time > 0:
            inputs = torch.cat(
                [
                    current_inventory,
                    past_expedited_orders[:, -expedited_lead_time:],
                ],
                dim=1,
            )
        else:
            inputs = current_inventory

        if regular_lead_time > 0:
            inputs = torch.cat(
                [inputs, past_regular_orders[:, -regular_lead_time:]], dim=1
            )
        return inputs

    def forward(self, inputs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.model is None:
            raise AttributeError("Model not initialized. Call `init_layers()` first.")

        h = self.model(inputs)
        # Prevent runaway order explosion into thousands of units (max single-period demand is <= 8)
        h = torch.clamp(h, min=0.0, max=40.0)
        q = h - torch.frac(h).detach()  # straight-through estimator

        # index 0: regular_q for period 0; indices 1..n_cycles: expedited_q per period
        return tuple(q[:, [i]] for i in range(self.n_cycles + 1))

    def predict(
        self,
        current_inventory: Union[int, torch.Tensor],
        past_regular_orders: Optional[Union[List[int], torch.Tensor]] = None,
        past_expedited_orders: Optional[Union[List[int], torch.Tensor]] = None,
        output_tensor: bool = False,
    ) -> Union[Tuple[torch.Tensor, torch.Tensor], Tuple[int, int]]:
        """
        Predict replenishment order quantities from the neural network.

        Parameters
        ----------
        current_inventory : int, or torch.Tensor
            Current inventory level.
        past_regular_orders : list, or torch.Tensor, optional
            Past regular orders. Padded or sliced to regular_lead_time.
        past_expedited_orders : list, or torch.Tensor, optional
            Past expedited orders. Padded or sliced to expedited_lead_time.
        output_tensor : bool, default is False
            If True, order quantities are returned as torch.Tensor; otherwise as integers.

        Returns
        -------
        tuple
            Tuple of (regular_order, expedited_order_0, expedited_order_1, ...).
        """
        if self.sourcing_model is None:
            raise AttributeError("The controller is not trained.")

        inputs = self.prepare_inputs(
            current_inventory,
            past_regular_orders,
            past_expedited_orders,
            sourcing_model=self.sourcing_model,
        )
        orders = self.forward(inputs)  # tuple of (n_cycles + 1) tensors

        if output_tensor:
            return orders
        else:
            return tuple(int(q.item()) for q in orders)


    def fit(
        self,
        sourcing_model: DualSourcingModel,
        sourcing_periods: int,
        epochs: int,
        validation_sourcing_periods: int = 1000,
        validation_freq: int = 50,
        log_freq: int = 10,
        init_inventory_freq: int = 4,
        init_inventory_lr: float = 1e-1,
        parameters_lr: float = 1e-4,
        seed: Optional[int] = None,
        checkpoint_path: Optional[str] = None,
        optimizer_type: str = 'rmsprop',   # 'rmsprop' (Böttcher et al.) | 'adam'
        use_scheduler: bool = False,       # optional LR decay
        use_grad_clip: bool = True,        # clip grad norm to 1.0
        device: str = 'cpu',               # 'cpu' | 'cuda'
        patience: int = 0,                 # early-stop: 0 = disabled
        target_vf: Optional[float] = None,
        vf_gap_tol: float = 0.005,
        vf_patience: int = 5,
    ) -> None:
        """
        Train the neural network controller using the sourcing model environment.

        Parameters
        ----------
        sourcing_model : DualSourcingModel
            The sourcing model environment for training.
        sourcing_periods : int
            Number of sourcing periods per training epoch.
        epochs : int
            Number of training epochs.
        validation_sourcing_periods : int, default 1000
            Number of sourcing periods for validation rollouts.
        validation_freq : int, default 50
            Interval of training epochs between validation evaluations.
        log_freq : int, default 10
            Interval of training epochs between logging.
        init_inventory_freq : int, default 4
            Interval of parameter epochs between initial inventory updates.
        init_inventory_lr : float, default 1e-1
            Learning rate for initial inventory parameter.
        parameters_lr : float, default 1e-4
            Learning rate for neural network weights.
        seed : int, optional
            Random seed for reproducibility.
        checkpoint_path : str, optional
            If provided, saves the best model checkpoint based on validation cost.
        optimizer_type : str, default 'rmsprop'
            Optimizer type: 'rmsprop' (Böttcher et al., alpha=0.99, eps=1e-8) or 'adam'.
        use_scheduler : bool, default False
            If True, applies CosineAnnealingLR decay.
        use_grad_clip : bool, default True
            If True, clips gradient norm to 1.0 to stabilize training on longer cycles.
        device : str, default 'cpu'
            Computation device ('cpu' or 'cuda').
        patience : int, default 0
            Patience for early stopping based on validation cost. If > 0, stops
            if validation cost does not improve for `patience` checks.
        target_vf : float, optional
            Certified DP value-function baseline for VF-aware early stopping.
        vf_gap_tol : float, default 0.005
            Relative distance to target_vf counting as within tolerance (0.005 = 0.5%).
        vf_patience : int, default 5
            Consecutive validation checks required within vf_gap_tol to trigger early stop.
        """

        assert optimizer_type in ('adam', 'rmsprop'), \
            f"optimizer_type must be 'adam' or 'rmsprop', got '{optimizer_type}'"
        assert validation_freq is not None, \
            "Validation frequency set to None, please provide an int value <= epochs"
        assert validation_freq <= epochs, \
            "Validation frequency > epochs, please provide an int value <= epochs"

        # ---- device setup -----------------------------------------------
        _device = torch.device(device)
        self.to(_device)
        sourcing_model.init_inventory.data = sourcing_model.init_inventory.data.to(_device)

        # Store sourcing model in self.sourcing_model
        self.sourcing_model = sourcing_model

        if seed is not None:
            torch.manual_seed(seed)

        if self.model is None:
            self.init_layers(
                regular_lead_time=sourcing_model.get_regular_lead_time(),
                expedited_lead_time=sourcing_model.get_expedited_lead_time(),
            )
            self.to(_device)

        start_time = datetime.now()
        logger.info(
            f"Sourcing periods are reduced by a factor of {self.n_cycles} "
            "to keep them aligned with other non-periodic controllers"
        )
        logger.info(
            f"Starting Multi-Period dual sourcing neural network training at {start_time}"
        )
        logger.info(
            f"Sourcing model parameters: batch_size={self.sourcing_model.batch_size}, "
            f"lead_time={self.sourcing_model.lead_time}, "
            f"init_inventory={self.sourcing_model.init_inventory.int().item()}, "
            f"demand_generator={self.sourcing_model.demand_generator.__class__.__name__}"
        )
        logger.info(
            f"Training parameters: epochs={epochs}, sourcing_periods={sourcing_periods}, "
            f"validation_cycles={validation_sourcing_periods}, "
            f"learning_rate={parameters_lr}, optimizer={optimizer_type}, "
            f"use_scheduler={use_scheduler}, use_grad_clip={use_grad_clip}, device={device}"
        )

        # ---- optimizers -------------------------------------------------
        optimizer_init_inventory = torch.optim.RMSprop(
            [sourcing_model.init_inventory], lr=init_inventory_lr
        )

        if optimizer_type == 'rmsprop':
            # Paper setup: RMSprop with alpha=0.99, eps=1e-8 (EC.5.1)
            optimizer_parameters = torch.optim.RMSprop(
                self.parameters(), lr=parameters_lr, alpha=0.99, eps=1e-8
            )
        else:
            # Alternative Adam optimizer
            optimizer_parameters = torch.optim.Adam(
                self.parameters(), lr=parameters_lr
            )

        # ---- optional scheduler -----------------------------------------
        scheduler = None
        if use_scheduler:
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer_parameters, T_max=epochs, eta_min=5e-5
            )

        min_loss = np.inf
        best_state = None   # will be set on first validation pass
        best_init_inventory = None
        N_VAL_SEEDS = 100   # 100-seed validation — much better proxy for EVAL_SEEDS=500
        no_improve_count = 0  # for early stopping
        vf_streak = 0          # consecutive validation checks within vf_gap_tol of target_vf

        for epoch in tqdm(range(epochs)):

            optimizer_init_inventory.zero_grad()
            optimizer_parameters.zero_grad()
            # get_total_cost handles reset and device placement internally
            train_loss = self.get_total_cost(sourcing_model, sourcing_periods)

            # NaN/Inf guard — stop immediately, don't waste more compute
            if not torch.isfinite(train_loss):
                tqdm.write(f"NaN/Inf training loss at epoch {epoch} — seed diverged, stopping early.")
                logger.warning("Diverged at epoch %d (train_loss=%s) — aborting seed.", epoch, train_loss.item())
                break

            train_loss.backward()

            if use_grad_clip:
                torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
                if sourcing_model.init_inventory.grad is not None:
                    torch.nn.utils.clip_grad_norm_([sourcing_model.init_inventory], max_norm=1.0)

            optimizer_init_inventory.step()
            # Prevent initial inventory from drifting to runaway negative or huge positive values
            sourcing_model.init_inventory.data.clamp_(min=0.0, max=30.0)
            optimizer_parameters.step()

            if scheduler is not None:
                scheduler.step()

            # Save the best model — average over N_VAL_SEEDS fixed seeds so the
            # comparison is deterministic and not biased by lucky demand draws.
            if epoch % validation_freq == 0:
                with torch.no_grad():
                    val_losses = []
                    for s in range(N_VAL_SEEDS):
                        # get_total_cost handles reset and device placement internally
                        val_losses.append(
                            self.get_total_cost(sourcing_model, validation_sourcing_periods, seed=s)
                        )
                eval_loss = torch.stack(val_losses).mean()
                val_avg = (eval_loss / validation_sourcing_periods).item()
                logger.info(
                    f"Epoch {epoch}/{epochs}"
                    f" - Validation cost: {val_avg:.4f}"
                )
                improved = eval_loss < min_loss
                if improved:
                    min_loss = eval_loss
                    best_state = {k: v.cpu() for k, v in self.state_dict().items()}
                    best_init_inventory = sourcing_model.init_inventory.item()
                    no_improve_count = 0
                    # Save immediately — checkpoint always reflects the best found so far
                    self.save_checkpoint(checkpoint_path, init_inventory=best_init_inventory)
                else:
                    no_improve_count += 1

                # ---- VF-aware stopping -------------------------------------------
                # best_val = (min_loss / periods) is monotonically non-increasing.
                # vf_streak counts consecutive checks where best_val is within
                # vf_gap_tol of VF AND the model did NOT improve this check.
                # This fires only when the model has both converged AND is near VF.
                if target_vf is not None:
                    best_val = (min_loss / validation_sourcing_periods).item()
                    gap = abs(best_val - target_vf) / abs(target_vf)
                    in_tol = gap <= vf_gap_tol
                    if in_tol and not improved:
                        vf_streak += 1   # stable inside VF tolerance → count toward stop
                    else:
                        vf_streak = 0    # not in tolerance yet, or still improving within it

                    if vf_streak >= vf_patience:
                        msg = (
                            f"VF-aware early stop at epoch {epoch}: "
                            f"best_val={best_val:.4f} within {vf_gap_tol*100:.1f}% "
                            f"of VF={target_vf:.4f} (GAP={gap*100:+.2f}%) "
                            f"— converged, stable for {vf_streak} checks."
                        )
                        tqdm.write(msg)
                        logger.info(msg)
                        break

                # ---- Patience (plateau) stopping ---------------------------------
                # Fires when validation has not improved for `patience` checks.
                # Case 1 (far from VF): bad local minimum — log warning.
                # Case 2 (near VF): VF-aware stop should have fired first if
                #   vf_patience < patience; otherwise this fires instead.
                if patience > 0 and no_improve_count >= patience:
                    best_val_final = (min_loss / validation_sourcing_periods).item()
                    if target_vf is not None:
                        gap_pct = (best_val_final - target_vf) / abs(target_vf) * 100
                        if abs(gap_pct) <= vf_gap_tol * 100:
                            verdict = f"near VF (GAP={gap_pct:+.2f}%)"
                            logger.info(
                                "Patience stop (near VF) epoch=%d best_val=%.4f "
                                "VF=%.4f GAP=%.2f%%",
                                epoch, best_val_final, target_vf, gap_pct,
                            )
                        else:
                            verdict = (
                                f"FAR from VF (GAP={gap_pct:+.2f}%) "
                                f"— bad local minimum, try different seed or lower LR"
                            )
                            logger.warning(
                                "Patience stop (far from VF) epoch=%d best_val=%.4f "
                                "VF=%.4f GAP=%.2f%%",
                                epoch, best_val_final, target_vf, gap_pct,
                            )
                        tqdm.write(
                            f"Early stop at epoch {epoch}: {verdict}  "
                            f"best_val={best_val_final:.4f} VF={target_vf:.4f}"
                        )
                    else:
                        tqdm.write(
                            f"Early stopping at epoch {epoch}: no improvement for "
                            f"{patience} checks ({patience * validation_freq} epochs)."
                        )
                        logger.info("Early stop epoch=%d patience=%d", epoch, patience)
                    break

            end_time = datetime.now()
            duration = end_time - start_time
            per_epoch_time = duration.total_seconds() / (epoch + 1)
            remaining_time = (epochs - epoch) * per_epoch_time
            if epoch % log_freq == 0:
                current_lr = optimizer_parameters.param_groups[0]['lr']
                logger.info(
                    f"Epoch {epoch}/{epochs}"
                    f" - Training cost: {train_loss / sourcing_periods:.4f}"
                    f" - lr: {current_lr:.6f}"
                    f" - Per epoch time: {per_epoch_time:.2f} seconds"
                    f" - Est. Remaining time: {int(remaining_time)} seconds."
                )

        # Restore best weights and best init inventory
        if best_state is not None:
            self.cpu()
            self.load_state_dict(best_state)
            if best_init_inventory is not None:
                sourcing_model.init_inventory.data.fill_(best_init_inventory)
        else:
            self.cpu()

        end_time = datetime.now()
        duration = end_time - start_time
        self.save_checkpoint(checkpoint_path, init_inventory=best_init_inventory)
        logger.info(f"Training completed at {end_time}")
        logger.info(f"Total training duration: {duration}")
        logger.info(
            f"Final best cost (avg over {N_VAL_SEEDS} seeds): "
            f"{min_loss / validation_sourcing_periods:.4f}"
        )

    def reset(self) -> None:
        """
        Reset the controller to the initial state.
        """
        self.model = None
        self.sourcing_model = None


    def get_last_cost(self, sourcing_model: DualSourcingModel) -> torch.Tensor:
        """Calculate the cost for the latest period."""
        last_regular_q = sourcing_model.get_last_regular_order()
        last_expedited_q = sourcing_model.get_last_expedited_order()
        regular_order_cost = sourcing_model.get_regular_order_cost()
        expedited_order_cost = sourcing_model.get_expedited_order_cost()
        holding_cost = sourcing_model.get_holding_cost()
        shortage_cost = sourcing_model.get_shortage_cost()
        current_inventory = sourcing_model.get_current_inventory()
        last_cost = (
            regular_order_cost * last_regular_q
            + expedited_order_cost * last_expedited_q
            + holding_cost * torch.relu(current_inventory)
            + shortage_cost * torch.relu(-current_inventory)
        )
        return last_cost

    def get_total_cost(
        self,
        sourcing_model: DualSourcingModel,
        sourcing_periods: int,
        seed: Optional[int] = None,
    ) -> torch.Tensor:
        """Calculate the total cost."""
        sourcing_model.reset()

        # Move sourcing model state tensors to the same device as the model.
        # reset() always allocates on CPU, so this is a no-op when device='cpu'.
        _dev = next(self.parameters()).device if len(list(self.parameters())) > 0 else torch.device('cpu')
        for _attr in ('past_inventories', 'past_demands',
                      'past_regular_orders', 'past_expedited_orders',
                      'past_orders'):
            if hasattr(sourcing_model, _attr):
                setattr(sourcing_model, _attr, getattr(sourcing_model, _attr).to(_dev))

        if seed is not None:
            torch.manual_seed(seed)

        # Accumulate on the model's device (no-op on CPU)
        total_cost = torch.tensor(0.0, device=_dev)

        for _ in range(sourcing_periods):
            current_inventory = sourcing_model.get_current_inventory()
            past_regular_orders = sourcing_model.get_past_regular_orders()
            past_expedited_orders = sourcing_model.get_past_expedited_orders()
            orders = self.predict(
                current_inventory,
                past_regular_orders,
                past_expedited_orders,
                output_tensor=True,
            )
            regular_q0    = orders[0]
            expedited_qs  = orders[1:]  # one per period in the cycle

            # Period 0: place regular + expedited order
            sourcing_model.order(regular_q0, expedited_qs[0])
            total_cost += self.get_last_cost(sourcing_model).mean()

            # Periods 1..n_cycles-1: no regular order, only expedited
            for expedited_q in expedited_qs[1:]:
                sourcing_model.order(torch.zeros_like(expedited_q), expedited_q)
                total_cost += self.get_last_cost(sourcing_model).mean()

        return total_cost

    def get_average_cost(
        self,
        sourcing_model: DualSourcingModel,
        sourcing_periods: int,
        seed: Optional[int] = None,
    ) -> torch.Tensor:
        """Calculate the average cost."""
        return (
            self.get_total_cost(sourcing_model, sourcing_periods, seed)
            / sourcing_periods
        )

    def save_checkpoint(self, path: str, init_inventory: Optional[float] = None) -> None:
        """Save model checkpoint including state dict and sourcing model config."""
        import os
        dirpath = os.path.dirname(path)
        if dirpath:
            os.makedirs(dirpath, exist_ok=True)
        inv = init_inventory if init_inventory is not None else self.sourcing_model.init_inventory.item()
        torch.save({
            'model_state_dict': self.state_dict(),
            'hidden_layers': self.hidden_layers,
            'n_cycles': self.n_cycles,
            'init_inventory': inv,
        }, path)
        logger.info(f"Checkpoint saved to {path}")

    @classmethod
    def load_checkpoint(
        cls,
        path: str,
        sourcing_model: DualSourcingModel,
        device: str = 'cpu',
    ) -> 'CyclicDualNeuralController':
        """
        Load a saved checkpoint for inference.
        """
        _dev = torch.device(device)
        checkpoint = torch.load(path, map_location=_dev)
        controller = cls(
            hidden_layers=checkpoint['hidden_layers'],
            n_cycles=checkpoint.get('n_cycles', 2),
        )
        controller.init_layers(
            regular_lead_time=sourcing_model.get_regular_lead_time(),
            expedited_lead_time=sourcing_model.get_expedited_lead_time(),
        )
        controller.load_state_dict(checkpoint['model_state_dict'])
        controller.to(_dev)
        controller.sourcing_model = sourcing_model
        
        # Restore trained initial inventory into the sourcing model
        if 'init_inventory' in checkpoint:
            sourcing_model.init_inventory.data.fill_(checkpoint['init_inventory'])
        sourcing_model.init_inventory.data = sourcing_model.init_inventory.data.to(_dev)
        
        logger.info(f"Checkpoint loaded from {path} onto {device}")
        return controller
