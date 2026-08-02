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
    Implements a multi-period nerual network architecture. Input consists of periodic time states
    E.g. I_t, I_(t+n), I_(t+2n)....
    Demand is realized internally for the whole cycle
    Cost calculation is based on the whole cycle
    """

    def __init__(
        self, 
        hidden_layers: List[int] = [128, 64, 32, 16, 8,4],
        activation: torch.nn.Module = torch.nn.CELU(alpha=1.0),
        n_cycles: int = 2
        ) -> None:

        """
        Parameters
        ----------
        hidden_layers: Architecure of hidden layers. hidden_layer[n] represents the number of nerons in layer n.
        activation: Activations betweeen layers
        n_cycles: Defines the number of time periods in 1 cycle. The output heads (and accordingly, the forward pass) will enumerate based on this value.
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
        #h = torch.clamp(h, min=0.0, max=20.0)
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
        Predict order qunatities from the neural network.

        Parameters
        ----------
        current_inventory : int, or torch.Tensor
            Current inventory.
        past_regular_orders : list, or torch.Tensor, optional
            Past regular orders. If the length of `past_regular_orders` is lower than `regular_lead_time`, it will be padded with zeros. If the length of `past_regular_orders` is higher than `regular_lead_time`, only the last `regular_lead_time` orders will be used during inference.
        past_expedited_orders : list, or torch.Tensor, optional
            Past expedited orders. If the length of `past_expedited_orders` is lower than `expedited_lead_time`, it will be padded with zeros. If the length of `past_expedited_orders` is higher than `expedited_lead_time`, only the last `expedited_lead_time` orders will be used during inference.
        output_tensor : bool, default is False
            If True, the replenishment order quantity will be returned as a torch.Tensor. Otherwise, it will be returned as an integer.

        Returns
        -------
        tuple
            A tuple containing the regular order quantity, expedited order quantity at time t, and expeditied order quantity at time t+1.
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
        # ----------------------------------------------------------------
        # Optional hyperparameter controls (Option A).
        # Defaults reproduce the colleague's original behaviour exactly so
        # pre_training.py / finetuning1.py continue to work unchanged.
        # Set optimizer_type='rmsprop', use_scheduler=False,
        # use_grad_clip=False to match the paper's setup.
        # ----------------------------------------------------------------
        optimizer_type: str = 'adam',      # 'adam' | 'rmsprop'
        use_scheduler: bool = True,        # cosine-annealing LR decay
        use_grad_clip: bool = True,        # clip grad norm to 1.0
        device: str = 'cpu',              # 'cpu' | 'cuda'
    ) -> None:
        """
        Train the neural network controller using the sourcing model and specified parameters.

        Parameters
        ----------
        sourcing_model : DualSourcingModel
            The sourcing model for training.
        sourcing_periods : int
            Number of sourcing periods for training.
        epochs : int
            Number of training epochs.
        validation_sourcing_periods : int, optional
            Number of sourcing periods for validation.
        validation_freq : int, default is 50
            Only relevant if `validation_sourcing_periods` is provided. Specifies how
            many training epochs to run before a new validation run is performed.
        log_freq : int, default is 10
            Specifies how many training epochs to run before logging the training cost.
        init_inventory_freq : int, default is 4
            Specifies how many parameter updating epochs to run before initial inventory
            is updated.
        init_inventory_lr : float, default is 1e-1
            Learning rate for initial inventory.
        parameters_lr : float, default is 1e-4
            Learning rate for updating neural network parameters.
        seed : int, optional
            Random seed for reproducibility.
        checkpoint_path : str, optional
            If provided, the best checkpoint (by validation cost) is saved here.
        optimizer_type : str, default 'adam'
            Which optimizer to use for NN parameters: 'adam' or 'rmsprop'.
            'rmsprop' matches the paper (Bottcher et al.) with alpha=0.99, eps=1e-8.
        use_scheduler : bool, default True
            If True, apply CosineAnnealingLR decay on top of the NN-parameters
            optimizer (colleague's addition).  Set False to match the paper.
        use_grad_clip : bool, default True
            If True, clip gradient norm to 1.0 before each step (colleague's
            addition).  Set False to match the paper.
        device : str, default 'cpu'
            Torch device string ('cpu' or 'cuda').  The model and all tensors
            produced during training are moved to this device.
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

        def _move_sm_to_device(sm, dev):
            """
            Move all state tensors of the sourcing model to `dev`.
            Called after every sm.reset() because reset() always allocates
            tensors on CPU regardless of the target device.
            """
            for attr in ('past_inventories', 'past_demands',
                         'past_regular_orders', 'past_expedited_orders',
                         'past_orders'):
                if hasattr(sm, attr):
                    setattr(sm, attr, getattr(sm, attr).to(dev))

        # Store sourcing model in self.sourcing_model
        self.sourcing_model = sourcing_model

        if seed is not None:
            torch.manual_seed(seed)

        if self.model is None:
            self.init_layers(
                regular_lead_time=sourcing_model.get_regular_lead_time(),
                expedited_lead_time=sourcing_model.get_expedited_lead_time(),
            )

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
            # Colleague's original setup
            optimizer_parameters = torch.optim.Adam(
                self.parameters(), lr=parameters_lr
            )

        # ---- optional scheduler -----------------------------------------
        scheduler = None
        if use_scheduler:
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer_parameters, T_max=epochs, eta_min=1e-5
            )

        min_loss = np.inf
        best_state = None   # will be set on first validation pass
        N_VAL_SEEDS = 10    # deterministic multi-seed validation

        for epoch in tqdm(range(epochs)):

            optimizer_init_inventory.zero_grad()
            optimizer_parameters.zero_grad()
            sourcing_model.reset()
            _move_sm_to_device(sourcing_model, _device)  # reset() puts tensors on CPU
            train_loss = self.get_total_cost(sourcing_model, sourcing_periods)
            train_loss.backward()

            if use_grad_clip:
                torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)

            optimizer_init_inventory.step()
            optimizer_parameters.step()

            if scheduler is not None:
                scheduler.step()

            # Save the best model — average over N_VAL_SEEDS fixed seeds so the
            # comparison is deterministic and not biased by lucky demand draws.
            if epoch % validation_freq == 0:
                with torch.no_grad():
                    val_losses = []
                    for s in range(N_VAL_SEEDS):
                        sourcing_model.reset()
                        _move_sm_to_device(sourcing_model, _device)
                        val_losses.append(
                            self.get_total_cost(sourcing_model, validation_sourcing_periods, seed=s)
                        )
                eval_loss = torch.stack(val_losses).mean()
                logger.info(
                    f"Epoch {epoch}/{epochs}"
                    f" - Validation cost: {eval_loss / validation_sourcing_periods:.4f}"
                )
                if eval_loss < min_loss:
                    min_loss = eval_loss
                    best_state = {k: v.cpu() for k, v in self.state_dict().items()}

            end_time = datetime.now()
            duration = end_time - start_time
            per_epoch_time = duration.total_seconds() / (epoch + 1)
            remaining_time = (epochs - epoch) * per_epoch_time
            if epoch % log_freq == 0:
                logger.info(
                    f"Epoch {epoch}/{epochs}"
                    f" - Training cost: {train_loss / sourcing_periods:.4f}"
                    f" - Per epoch time: {per_epoch_time:.2f} seconds"
                    f" - Est. Remaining time: {int(remaining_time)} seconds."
                )

        # Restore best weights (always on CPU for portability)
        if best_state is not None:
            self.cpu()
            self.load_state_dict(best_state)
        else:
            self.cpu()

        end_time = datetime.now()
        duration = end_time - start_time
        self.save_checkpoint(checkpoint_path)
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

    def save_checkpoint(self, path: str) -> None:
        """Save model checkpoint including state dict and sourcing model config."""
        import os
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save({
            'model_state_dict': self.state_dict(),
            'hidden_layers': self.hidden_layers,
            'n_cycles': self.n_cycles,
            'init_inventory': self.sourcing_model.init_inventory.item(),
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

        Parameters
        ----------
        path : str
            Path to the .pt checkpoint file.
        sourcing_model : DualSourcingModel
            The sourcing model to attach to the controller.
        device : str, default 'cpu'
            Device to load the model onto ('cpu' or 'cuda').
            Pass the same device used during training so evaluation
            runs on GPU rather than CPU.
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
        sourcing_model.init_inventory.data = sourcing_model.init_inventory.data.to(_dev)
        logger.info(f"Checkpoint loaded from {path} onto {device}")
        return controller