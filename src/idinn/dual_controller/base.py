from abc import ABCMeta, abstractmethod
from typing import List, Optional, Tuple, Union, no_type_check

import numpy as np
import torch
from matplotlib import pyplot as plt
from numpy.typing import NDArray

from ..sourcing_model import DualSourcingModel


class BaseDualController(metaclass=ABCMeta):
    @abstractmethod
    def fit(self, sourcing_model: DualSourcingModel, **kwargs) -> None:
        """
        Fit the controller to the sourcing model.
        """
        pass

    @abstractmethod
    def predict(
        self,
        current_inventory: Union[int, torch.Tensor],
        past_regular_orders: Optional[Union[List[int], torch.Tensor]] = None,
        past_expedited_orders: Optional[Union[List[int], torch.Tensor]] = None,
        output_tensor: bool = False,
    ) -> Union[Tuple[torch.Tensor, torch.Tensor], Tuple[int, int]]:
        """
        Predict the replenishment order quantity.
        """
        pass

    @abstractmethod
    def reset(self) -> None:
        """
        Reset the controller to the initial state.
        """
        pass

    def _check_current_inventory(
        self, current_inventory: Union[int, torch.Tensor]
    ) -> torch.Tensor:
        """Check and convert types of current_inventory."""
        if isinstance(current_inventory, int):
            return torch.tensor([[current_inventory]], dtype=torch.float32)
        elif isinstance(current_inventory, torch.Tensor):
            return current_inventory
        raise TypeError("`current_inventory`'s type is not supported.")

    def _check_past_orders(
        self, past_orders: Optional[Union[List[int], torch.Tensor]], lead_time: int
    ) -> torch.Tensor:
        """Check and convert types of past orders."""
        if past_orders is None:
            past_orders = torch.zeros(1, lead_time)
        elif isinstance(past_orders, list):
            past_orders = torch.tensor([past_orders], dtype=torch.float32)
        elif isinstance(past_orders, torch.Tensor):
            pass
        else:
            raise TypeError("`past_orders`'s type is not supported.")

        order_len = past_orders.shape[1]
        if order_len < lead_time:
            return torch.nn.functional.pad(past_orders, (lead_time - order_len, 0))
        else:
            return past_orders

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

    @no_type_check
    def get_total_cost(
        self,
        sourcing_model: DualSourcingModel,
        sourcing_periods: int,
        seed: Optional[int] = None,
    ) -> torch.Tensor:
        """Calculate the total cost."""
        if seed is not None:
            torch.manual_seed(seed)

        total_cost = torch.tensor(0.0)
        for _ in range(sourcing_periods):
            current_inventory = sourcing_model.get_current_inventory()
            past_regular_orders = sourcing_model.get_past_regular_orders()
            past_expedited_orders = sourcing_model.get_past_expedited_orders()
            regular_q, expedited_q = self.predict(
                current_inventory,
                past_regular_orders,
                past_expedited_orders,
                output_tensor=True,
            )
            sourcing_model.order(regular_q, expedited_q)
            last_cost = self.get_last_cost(sourcing_model)
            total_cost += last_cost.mean()
        return total_cost

    @no_type_check
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

    @no_type_check
    def simulate(
        self,
        sourcing_model: DualSourcingModel,
        sourcing_periods: int,
        seed: Optional[int] = None,
    ) -> Tuple[NDArray[np.float32], NDArray[np.float32], NDArray[np.float32]]:
        """Simulate the sourcing model's output."""
        if seed is not None:
            torch.manual_seed(seed)
        sourcing_model.reset(batch_size=1)
        for i in range(sourcing_periods):
            current_inventory = sourcing_model.get_current_inventory()
            past_regular_orders = sourcing_model.get_past_regular_orders()
            past_expedited_orders = sourcing_model.get_past_expedited_orders()
            regular_q, expedited_q = self.predict(
                current_inventory, past_regular_orders, past_expedited_orders
            )
            sourcing_model.order(regular_q, expedited_q)
        past_inventories = (
            sourcing_model.get_past_inventories()[0, :].detach().cpu().numpy()
        )
        past_regular_orders = (
            sourcing_model.get_past_regular_orders()[0, :].detach().cpu().numpy()
        )
        past_expedited_orders = (
            sourcing_model.get_past_expedited_orders()[0, :].detach().cpu().numpy()
        )
        return past_inventories, past_regular_orders, past_expedited_orders

    def plot(
        self,
        sourcing_model: DualSourcingModel,
        sourcing_periods: int,
        linewidth: int = 1,
        seed: Optional[int] = None,
    ) -> Tuple[plt.Figure, plt.Axes]:
        """Plot the inventory and order quantities."""
        past_inventories, past_regular_orders, past_expedited_orders = self.simulate(
            sourcing_model=sourcing_model, sourcing_periods=sourcing_periods, seed=seed
        )
        fig, ax = plt.subplots(ncols=2, figsize=(10, 4))
        ax[0].step(
            range(sourcing_periods),
            past_inventories[-sourcing_periods:],
            linewidth=linewidth,
            color="tab:blue",
        )
        ax[0].yaxis.get_major_locator().set_params(integer=True)
        ax[0].set_title("Inventory")
        ax[0].set_xlabel("Period")
        ax[0].set_ylabel("Quantity")

        ax[1].step(
            range(sourcing_periods),
            past_expedited_orders[-sourcing_periods:],
            label="Expedited Order",
            linewidth=linewidth,
            color="tab:green",
        )
        ax[1].step(
            range(sourcing_periods),
            past_regular_orders[-sourcing_periods:],
            label="Regular Order",
            linewidth=linewidth,
            color="tab:orange",
        )
        ax[1].yaxis.get_major_locator().set_params(integer=True)
        ax[1].set_title("Order")
        ax[1].set_xlabel("Period")
        ax[1].set_ylabel("Quantity")
        ax[1].legend()

        return fig, ax
