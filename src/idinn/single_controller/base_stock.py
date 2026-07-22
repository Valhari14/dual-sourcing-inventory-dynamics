import logging
from datetime import datetime
from typing import List, Optional, Union, no_type_check

import torch

from ..sourcing_model import SingleSourcingModel
from .base import BaseSingleController

# Add logger setup at class level
logger = logging.getLogger(__name__)


class BaseStockController(BaseSingleController):
    """
    Base stock controller for single-sourcing inventory optimization.
    """

    def __init__(self) -> None:
        self.sourcing_model: Optional[SingleSourcingModel] = None
        self.z_star: Optional[int] = None
        logger.info("Initialized `BaseStockController`")

    @no_type_check
    def fit(
        self,
        sourcing_model: SingleSourcingModel,
        num_samples: int = 100000,
        seed: Optional[int] = None,
    ) -> None:
        """
        Calculate the optimal target inventory level z* and store it in self.z_star.

        Parameters
        ----------
        sourcing_model : SourcingModel
            The sourcing model for training.
        num_samples : int
            The number of samples used for calculating empirical percentile.
        seed : int, optional
            Random seed for reproducibility.
        """
        # Store sourcing model in self.sourcing_model
        self.sourcing_model = sourcing_model

        if seed is not None:
            torch.manual_seed(seed)

        start_time = datetime.now()
        logger.info(f"Starting base stock policy calculation at {start_time}")
        logger.info(
            f"Sourcing model parameters: batch_size={self.sourcing_model.batch_size}, "
            f"lead_time={self.sourcing_model.lead_time}, init_inventory={self.sourcing_model.init_inventory.int().item()}, "
            f"demand_generator={self.sourcing_model.demand_generator.__class__.__name__}"
        )
        logger.info(f"Training parameters: num_samples={num_samples}")

        # Generate samples for l + 1 periods
        samples = sourcing_model.demand_generator.sample(
            batch_size=num_samples, batch_width=self.sourcing_model.get_lead_time() + 1
        )

        # Calculate the total demand for each sample
        total_demand_samples = samples.sum(dim=1)

        # Get shortage cost and holding cost from sourcing model
        b = sourcing_model.get_shortage_cost()
        h = sourcing_model.get_holding_cost()

        # Calculate z* using the empirical percentile (inverse CDF)
        service_level = b / (b + h)
        self.z_star = int(
            torch.quantile(total_demand_samples.float(), service_level).item()
        )

        end_time = datetime.now()
        duration = end_time - start_time
        logger.info(f"Policy calculation completed at {end_time}")
        logger.info(f"Total calculation duration: {duration}")
        logger.info(f"Optimal base stock level (z*): {self.z_star}")
        logger.info(
            f"Final best cost: {self.get_average_cost(self.sourcing_model, sourcing_periods=1000, seed=42):.4f}"
        )

    def predict(
        self,
        current_inventory: Union[int, torch.Tensor],
        past_orders: Optional[Union[List[int], torch.Tensor]] = None,
        output_tensor: bool = False,
    ) -> Union[torch.Tensor, int]:
        """
        Calculate the replenishment order quantity.

        Parameters
        ----------
        current_inventory : int
            Current inventory level.
        past_orders : list, or torch.Tensor, optional
            Array of past orders. If `past_orders` is None, or the length of `past_orders` is lower than `lead_time`, it will be padded with zeros. If the length of `past_orders` is higher than `lead_time`, only the last `lead_time` orders will be used during inference.
        output_tensor : bool, default is False
            If True, the replenishment order quantity will be returned as a torch.Tensor. Otherwise, it will be returned as an integer.

        Returns
        -------
        float
            The replenishment order quantity.
        """
        if self.sourcing_model is None or self.z_star is None:
            raise AttributeError("The controller is not trained.")

        lead_time = self.sourcing_model.get_lead_time()
        current_inventory = self._current_inventory_check(current_inventory)
        past_orders = self._past_orders_check(lead_time, past_orders)

        if lead_time == 0:
            inventory_position = current_inventory
        elif lead_time > 0:
            inventory_position = current_inventory + past_orders[:, -lead_time:].sum(
                dim=1, keepdim=True
            )
        else:
            raise ValueError("`lead_time` cannot be less than 0")

        result = torch.relu(self.z_star - inventory_position)

        if output_tensor:
            return result
        else:
            return int(result.item())

    def reset(self) -> None:
        self.z_star = None
        self.sourcing_model = None
