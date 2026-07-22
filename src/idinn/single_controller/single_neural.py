import logging
from datetime import datetime
from typing import List, Optional, Union, no_type_check

import torch
from torch.utils.tensorboard import SummaryWriter

from ..sourcing_model import SingleSourcingModel
from .base import BaseSingleController

# Add logger setup at class level
logger = logging.getLogger(__name__)


class SingleSourcingNeuralController(torch.nn.Module, BaseSingleController):
    """
    SingleSourcingNeuralController is a neural network-based controller for inventory optimization in a single-sourcing scenario.

    Parameters
    ----------
    hidden_layers : list, default is [2]
        List of integers representing the number of units in each hidden layer.
    activation : torch.nn.Module, default is torch.nn.CELU(alpha=1)
        Activation function to be used in the hidden layers.

    Attributes
    ----------
    hidden_layers : list
        List of integers representing the number of units in each hidden layer.
    activation : torch.nn.Module
        Activation function used in the hidden layers.
    architecture : torch.nn.Sequential
        Sequential stack of linear layers and activation functions.

    Methods
    -------
    init_layers(lead_time)
        Initialize the layers of the neural network based on the lead time.
    forward(current_inventory, past_orders)
        Perform forward pass through the neural network.
    get_total_cost(sourcing_model, sourcing_periods, seed=None)
        Calculate the total cost over a given number of sourcing periods.
    train(sourcing_model, sourcing_periods, epochs, ...)
        Train the neural network controller using the sourcing model and specified parameters.
    simulate(sourcing_model, sourcing_periods)
        Simulate the inventory and order quantities over a given number of sourcing periods.
    plot(sourcing_model, sourcing_periods)
        Plot the inventory and order quantities over a given number of sourcing periods.
    """

    def __init__(
        self,
        hidden_layers: List[int] = [2],
        activation: torch.nn.Module = torch.nn.CELU(alpha=1),
    ):
        super().__init__()
        self.sourcing_model: Optional[SingleSourcingModel] = None
        self.hidden_layers = hidden_layers
        self.activation = activation
        self.model: Optional[torch.nn.Sequential] = None
        logger.info(
            f"Initialized SingleSourcingNeuralController with hidden_layers={hidden_layers}"
        )

    def init_layers(self) -> None:
        """
        Initialize the layers of the neural network based on the lead time.

        Returns
        -------
        None
        """
        if self.sourcing_model is None:
            raise AttributeError("The controller is not trained.")

        lead_time = self.sourcing_model.get_lead_time()
        architecture = [
            torch.nn.Linear(lead_time + 1, self.hidden_layers[0]),
            self.activation,
        ]
        for i in range(len(self.hidden_layers)):
            if i < len(self.hidden_layers) - 1:
                architecture += [
                    torch.nn.Linear(self.hidden_layers[i], self.hidden_layers[i + 1]),
                    self.activation,
                ]
        architecture += [
            torch.nn.Linear(self.hidden_layers[-1], 1, bias=False),
            torch.nn.ReLU(),
        ]
        self.model = torch.nn.Sequential(*architecture)

    def prepare_inputs(
        self,
        current_inventory: Union[int, torch.Tensor],
        past_orders: Optional[Union[List[int], torch.Tensor]] = None,
    ) -> torch.Tensor:
        if self.sourcing_model is None:
            raise AttributeError("The controller is not trained.")

        lead_time = self.sourcing_model.get_lead_time()
        current_inventory = self._current_inventory_check(current_inventory)
        past_orders = self._past_orders_check(lead_time, past_orders)

        if lead_time == 0:
            return current_inventory
        elif lead_time > 0:
            if past_orders is None:
                past_orders = torch.zeros(current_inventory.shape[0], lead_time)
            inputs = torch.cat([current_inventory, past_orders[:, -lead_time:]], dim=1)
        else:
            raise ValueError("`lead_time` cannot be less than 0")

        return inputs

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        if self.model is None:
            raise AttributeError("Model not initialized. Call `init_layers()` first.")
        h = self.model(inputs)
        q = h - torch.frac(h).clone().detach()
        return q

    def predict(
        self,
        current_inventory: Union[int, torch.Tensor],
        past_orders: Optional[Union[List[int], torch.Tensor]] = None,
        output_tensor: bool = False,
    ) -> Union[torch.Tensor, int]:
        """
        Perform forward pass through the neural network.

        Parameters
        ----------
        current_inventory : int, or torch.Tensor
            Current inventory levels.
        past_orders : list, or torch.Tensor, optional
            Past order quantities. If the length of `past_orders` is lower than `lead_time`, it will be padded with zeros. If the length of `past_orders` is higher than `lead_time`, only the last `lead_time` orders will be used during inference.
        output_tensor : bool, default is False
            If True, the replenishment order quantity will be returned as a torch.Tensor. Otherwise, it will be returned as an integer.

        Returns
        -------
        torch.Tensor
            Order quanty calculated by the neural network.
        """
        if self.sourcing_model is None:
            raise AttributeError("The controller is not trained.")

        inputs = self.prepare_inputs(current_inventory, past_orders)
        q = self.forward(inputs)

        if output_tensor:
            return q
        return int(q.item())

    @no_type_check
    def fit(
        self,
        sourcing_model: SingleSourcingModel,
        sourcing_periods: int,
        epochs: int,
        validation_sourcing_periods: Optional[int] = None,
        validation_freq: int = 50,
        log_freq: int = 50,
        init_inventory_lr: float = 1e-1,
        init_inventory_freq: int = 4,
        parameters_lr: float = 3e-3,
        tensorboard_writer: Optional[SummaryWriter] = None,
        seed: Optional[int] = None,
    ) -> None:
        """
        Train the neural network controller using the sourcing model and specified parameters.

        Parameters
        ----------
        sourcing_model : SourcingModel
            The sourcing model for training.
        sourcing_periods : int
            The number of sourcing periods for training.
        epochs : int
            The number of training epochs.
        validation_sourcing_periods : int, optional
            The number of sourcing periods for validation.
        validation_freq : int, default is 10
            Only relevant if `validation_sourcing_periods` is provided. Specifies how many training epochs to run before a new validation run is performed, e.g. `validation_freq=10` runs validation every 10 epochs.
        log_freq : int, default is 10
            Specifies how many training epochs to run before logging the training loss.
        init_inventory_freq : int, default is 4
            Specifies how many parameter updating epochs to run before initial inventory is updated. e.g. `init_inventory_freq=4` updates initial inventory after updating parameters for 4 epochs.
        init_inventory_lr : float, default is 1e-1
            Learning rate for initial inventory.
        parameters_lr : float, default is 3e-3
            Learning rate for updating neural network parameters.
        tensorboard_writer : tensorboard.SummaryWriter, optional
            Tensorboard writer for logging.
        seed : int, optional
            Random seed for reproducibility.
        """
        # Store sourcing model in self.sourcing_model
        self.sourcing_model = sourcing_model

        if seed is not None:
            torch.manual_seed(seed)

        if self.model is None:
            self.init_layers()

        start_time = datetime.now()
        logger.info(f"Starting single sourcing neural network training at {start_time}")
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
        min_cost: torch.Tensor = torch.inf
        best_state: Optional[dict] = None

        for epoch in range(epochs):
            # Clear grad cache
            optimizer_parameters.zero_grad()
            optimizer_init_inventory.zero_grad()
            # Reset the sourcing model with the learned init inventory
            sourcing_model.reset()
            logger.debug(
                f"Reset the sourcing model with the learned init inventory at epoch {epoch}"
            )
            total_cost = self.get_total_cost(sourcing_model, sourcing_periods)
            total_cost.backward()
            # Gradient descend
            if epoch % init_inventory_freq == 0:
                optimizer_init_inventory.step()
            else:
                optimizer_parameters.step()
            # Save the best model
            if validation_sourcing_periods is not None and epoch % 10 == 0:
                eval_cost = self.get_total_cost(
                    sourcing_model, validation_sourcing_periods
                )
                if eval_cost < min_cost:
                    min_cost = eval_cost
                    if self.model is not None:
                        best_state = self.model.state_dict()
            else:
                if total_cost < min_cost:
                    min_cost = total_cost
                    best_state = self.model.state_dict()
            # Log train loss
            if tensorboard_writer is not None:
                tensorboard_writer.add_scalar(
                    "Avg. cost per period/train", total_cost / sourcing_periods, epoch
                )
                if (
                    validation_sourcing_periods is not None
                    and epoch % validation_freq == 0
                ):
                    # Log validation loss
                    eval_cost = self.get_total_cost(
                        sourcing_model, validation_sourcing_periods
                    )
                    tensorboard_writer.add_scalar(
                        "Avg. cost per period/val",
                        eval_cost / validation_sourcing_periods,
                        epoch,
                    )
                logger.debug(f"Wrote to tensorboard at epoch {epoch}")
                tensorboard_writer.flush()

            end_time = datetime.now()
            duration = end_time - start_time
            per_epoch_time = duration.total_seconds() / (epoch + 1)  # seconds per epoch
            remaining_time = (epochs - epoch) * per_epoch_time

            if epoch % log_freq == 0:
                logger.info(
                    f"Epoch {epoch}/{epochs}"
                    f" - Training cost: {total_cost / sourcing_periods:.4f}"
                    f" - Per epoch time: {per_epoch_time:.2f} seconds"
                    f" - Est. Remaining time: {int(remaining_time)} seconds."
                )

            if validation_sourcing_periods is not None and epoch % validation_freq == 0:
                logger.info(
                    f"Epoch {epoch}/{epochs}"
                    f" - Validation cost: {eval_cost / validation_sourcing_periods:.4f}"
                    f" - Per epoch time: {per_epoch_time:.2f} seconds"
                    f" - Est. Remaining time: {int(remaining_time)} seconds."
                )

        # Load the best model
        if best_state is not None and self.model is not None:
            self.model.load_state_dict(best_state)

        end_time = datetime.now()
        duration = end_time - start_time
        logger.info(f"Training completed at {end_time}")
        logger.info(f"Total training duration: {duration}")
        logger.info(f"Final best cost: {min_cost / sourcing_periods:.4f}")

    def reset(self) -> None:
        self.sourcing_model = None
        self.model = None

    def save(self, path: str) -> None:
        torch.save(self.model, path)

    def load(self, path: str) -> None:
        self.model = torch.load(path, weights_only=False)
