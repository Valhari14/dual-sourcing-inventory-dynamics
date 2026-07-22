import logging
from datetime import datetime
from typing import List, Optional, Tuple, Union, no_type_check

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

from ..sourcing_model import DualSourcingModel
from .base import BaseDualController

# Get root logger
logger = logging.getLogger()


class DualSourcingNeuralController(torch.nn.Module, BaseDualController):
    """
    DualSourcingNeuralController is a neural network controller for dual sourcing inventory optimization.

    Parameters
    ----------
    hidden_layers : list, default is [128, 64, 32, 16, 8, 4]
        List of integers specifying the sizes of hidden layers.
    activation : torch.nn.Module, default is torch.nn.CELU(alpha=1)
        Activation function to be used in the hidden layers.
    compressed : bool, default is False
        Flag indicating whether the input is compressed.

    Attributes
    ----------
    hidden_layers : list
        List of integers specifying the sizes of hidden layers.
    activation : torch.nn.Module
        Activation function to be used in the hidden layers.
    compressed : bool
        Flag indicating whether the input is compressed.
    regular_lead_time : int
        Regular lead time.
    expedited_lead_time : int
        Expedited lead time.
    architecture : torch.nn.Sequential
        Sequential stack of linear layers and activation functions.

    Methods
    -------
    init_layers(regular_lead_time, expedited_lead_time)
        Initialize the layers of the neural network.
    forward(current_inventory, past_orders)
        Forward pass of the neural network.
    get_total_cost(sourcing_model, sourcing_periods, seed=None)
        Calculate the total cost of the sourcing model.
    train(sourcing_model, sourcing_periods, epochs, ...)
        Trains the neural network controller using the sourcing model and specified parameters.
    simulate(sourcing_model, sourcing_periods, seed=None)
        Simulate the sourcing model using the neural network.
    plot(sourcing_model, sourcing_periods)
        Plot the inventory and order quantities.
    """

    def __init__(
        self,
        hidden_layers: List[int] = [128, 64, 32, 16, 8, 4],
        activation: torch.nn.Module = torch.nn.ReLU(),
        compressed: bool = False,
    ):
        super().__init__()
        self.sourcing_model = None
        self.hidden_layers = hidden_layers
        self.activation = activation
        self.compressed = compressed
        self.model: Optional[torch.nn.Sequential] = None
        logger.info(
            f"Initialized DualSourcingNeuralController with hidden_layers={hidden_layers}, compressed={compressed}"
        )

    def init_layers(self, regular_lead_time: int, expedited_lead_time: int) -> None:
        """
        Initialize the layers of the neural network.

        Parameters
        ----------
        regular_lead_time : int
            Regular lead time.
        expedited_lead_time : int
            Expedited lead time.
        """
        if self.compressed:
            input_length = regular_lead_time + expedited_lead_time
        else:
            input_length = regular_lead_time + expedited_lead_time + 1

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
            torch.nn.Linear(self.hidden_layers[-1], 2),
            # TODO: Mention this ReLU layer in documentation
            torch.nn.ReLU(),
        ]
        self.model = torch.nn.Sequential(*architecture)
        logger.info(
            f"Initialized neural network layers with regular_lead_time={regular_lead_time}, "
            f"expedited_lead_time={expedited_lead_time}"
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

        if regular_lead_time > 0:
            if self.compressed:
                inputs = past_regular_orders[:, -regular_lead_time:]
                inputs[:, 0] += current_inventory
            else:
                inputs = torch.cat(
                    [
                        current_inventory,
                        past_regular_orders[:, -regular_lead_time:],
                    ],
                    dim=1,
                )
        else:
            inputs = current_inventory

        if expedited_lead_time > 0:
            inputs = torch.cat(
                [inputs, past_expedited_orders[:, -expedited_lead_time:]], dim=1
            )
        return inputs

    def forward(self, inputs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.model is None:
            raise AttributeError("Model not initialized. Call `init_layers()` first.")

        h = self.model(inputs)
        q = h - torch.frac(h).clone().detach()
        regular_q = q[:, [0]]
        expedited_q = q[:, [1]]
        return regular_q, expedited_q

    def predict(
        self,
        current_inventory: Union[int, torch.Tensor],
        past_regular_orders: Optional[Union[List[int], torch.Tensor]] = None,
        past_expedited_orders: Optional[Union[List[int], torch.Tensor]] = None,
        output_tensor: bool = False,
    ) -> Union[Tuple[torch.Tensor, torch.Tensor], Tuple[int, int]]:
        """
        Forward pass of the neural network.

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
            A tuple containing the regular order quantity and expedited order quantity.
        """
        if self.sourcing_model is None:
            raise AttributeError("The controller is not trained.")

        inputs = self.prepare_inputs(
            current_inventory,
            past_regular_orders,
            past_expedited_orders,
            sourcing_model=self.sourcing_model,
        )
        regular_q, expedited_q = self.forward(inputs)

        if output_tensor:
            return regular_q, expedited_q
        else:
            return int(regular_q), int(expedited_q)

    @no_type_check
    def fit(
        self,
        sourcing_model: DualSourcingModel,
        sourcing_periods: int,
        epochs: int,
        validation_sourcing_periods: Optional[int] = None,
        validation_freq: int = 50,
        log_freq: int = 100,
        init_inventory_freq: int = 4,
        init_inventory_lr: float = 1e-1,
        parameters_lr: float = 3e-3,
        tensorboard_writer: Optional[SummaryWriter] = None,
        seed: Optional[int] = None,
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
        validation_freq : int, default is 10
            Only relevant if `validation_sourcing_periods` is provided. Specifies how many training epochs to run before a new validation run is performed, e.g. `validation_freq=10` runs validation every 10 epochs.
        log_freq : int, default is 10
            Specifies how many training epochs to run before logging the training cost.
        init_inventory_freq : int, default is 4
            Specifies how many parameter updating epochs to run before initial inventory is updated. e.g. `init_inventory_freq=4` updates initial inventory after updating parameters for 4 epochs.
        init_inventory_lr : float, default is 1e-1
            Learning rate for initial inventory.
        parameters_lr : float, default is 3e-3
            Learning rate for updating neural network parameters.
        tensorboard_writer : tensorboard.SummaryWriter, optional
        seed : int, optional
            Random seed for reproducibility.
            Tensorboard writer for logging.
        """
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
        logger.info(f"Starting dual sourcing neural network training at {start_time}")
        logger.info(
            f"Sourcing model parameters: batch_size={self.sourcing_model.batch_size}, "
            f"lead_time={self.sourcing_model.lead_time}, init_inventory={self.sourcing_model.init_inventory.int().item()}, "
            f"demand_generator={self.sourcing_model.demand_generator.__class__.__name__}"
        )
        logger.info(
            f"Training parameters: epochs={epochs}, sourcing_periods={sourcing_periods}, "
            f"validation_periods={validation_sourcing_periods}, learning_rate={parameters_lr}"
        )

        optimizer_init_inventory = torch.optim.RMSprop(
            [sourcing_model.init_inventory], lr=init_inventory_lr
        )
        optimizer_parameters = torch.optim.RMSprop(self.parameters(), lr=parameters_lr)
        min_loss = np.inf

        for epoch in range(epochs):
            # Clear grad cache
            optimizer_init_inventory.zero_grad()
            optimizer_parameters.zero_grad()
            # Reset the sourcing model with the learned init inventory
            sourcing_model.reset()
            train_loss = super().get_total_cost(sourcing_model, sourcing_periods)
            train_loss.backward()
            # Perform gradient descend
            if epoch % init_inventory_freq == 0:
                optimizer_init_inventory.step()
            else:
                optimizer_parameters.step()
            # Save the best model
            if validation_sourcing_periods is not None and epoch % validation_freq == 0:
                eval_loss = super().get_total_cost(
                    sourcing_model, validation_sourcing_periods
                )
                if eval_loss < min_loss:
                    min_loss = eval_loss
                    best_state = self.state_dict()
            else:
                if train_loss < min_loss:
                    min_loss = train_loss
                    best_state = self.state_dict()
            # Log train loss
            if tensorboard_writer is not None:
                tensorboard_writer.add_scalar(
                    "Avg. cost per period/train", train_loss / sourcing_periods, epoch
                )
                if validation_sourcing_periods is not None and epoch % 10 == 0:
                    # Log validation loss
                    tensorboard_writer.add_scalar(
                        "Avg. cost per period/val",
                        eval_loss / validation_sourcing_periods,
                        epoch,
                    )
                tensorboard_writer.flush()

            end_time = datetime.now()
            duration = end_time - start_time
            per_epoch_time = duration.total_seconds() / (epoch + 1)  # seconds per epoch
            remaining_time = (epochs - epoch) * per_epoch_time
            if epoch % log_freq == 0:
                logger.info(
                    f"Epoch {epoch}/{epochs}"
                    f" - Training cost: {train_loss / sourcing_periods:.4f}"
                    f" - Per epoch time: {per_epoch_time:.2f} seconds"
                    f" - Est. Remaining time: {int(remaining_time)} seconds."
                )

            if validation_sourcing_periods is not None and epoch % validation_freq == 0:
                logger.info(
                    f"Epoch {epoch}/{epochs}"
                    f" - Validation cost: {eval_loss / validation_sourcing_periods:.4f}"
                    f" - Per epoch time: {per_epoch_time:.2f} seconds"
                    f" - Est. Remaining time: {int(remaining_time)} seconds."
                )

        self.load_state_dict(best_state)

        end_time = datetime.now()
        duration = end_time - start_time
        logger.info(f"Training completed at {end_time}")
        logger.info(f"Total training duration: {duration}")
        logger.info(f"Final best cost: {min_loss / sourcing_periods:.4f}")

    def reset(self) -> None:
        """
        Reset the controller to the initial state.
        """
        self.model = None
        self.sourcing_model = None

    def save(self, path: str) -> None:
        torch.save(self.model, path)

    def load(self, path: str) -> None:
        self.model = torch.load(path, weights_only=False)
