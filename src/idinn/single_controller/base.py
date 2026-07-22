from abc import ABCMeta, abstractmethod
from typing import List, Optional, Union, no_type_check

import torch
from matplotlib import pyplot as plt
from numpy.typing import NDArray

from ..sourcing_model import SingleSourcingModel


class BaseSingleController(metaclass=ABCMeta):
    @abstractmethod
    def fit(
        self,
        sourcing_model: SingleSourcingModel,
        **kwargs,
    ) -> None:
        """
        Fit the controller to the sourcing model.
        """
        pass

    @abstractmethod
    def predict(
        self,
        current_inventory: Union[int, torch.Tensor],
        past_orders: Union[List[int], torch.Tensor],
        output_tensor: bool,
    ) -> Union[torch.Tensor, int]:
        """
        Predict the replenishment order quantity.
        """
        pass

    @abstractmethod
    def reset(self):
        """
        Reset the controller to the initial state.
        """
        pass

    def _current_inventory_check(
        self, current_inventory: Union[int, torch.Tensor]
    ) -> torch.Tensor:
        """
        Check and convert types of `current_inventory` for `predict()`.
        """
        if isinstance(current_inventory, int):
            current_inventory = torch.tensor([[current_inventory]], dtype=torch.float32)
        elif isinstance(current_inventory, torch.Tensor):
            pass
        else:
            raise TypeError("`current_inventory`'s type is not supported.")

        return current_inventory

    def _past_orders_check(
        self,
        lead_time: int,
        past_orders: Optional[Union[List[int], torch.Tensor]] = None,
    ) -> torch.Tensor:
        """
        Check and convert types of `past_orders` for `predict()`. Pad `past_orders` with zeros if it is too short.
        """
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

    def get_last_cost(self, sourcing_model: SingleSourcingModel) -> torch.Tensor:
        """
        Calculate the cost for the latest period of the sourcing model.

        Parameters
        ----------
        sourcing_model : SingleSourcingModel
            The sourcing model.

        Returns
        -------
        torch.Tensor
            The last cost.

        """
        shortage_cost = sourcing_model.get_shortage_cost()
        holding_cost = sourcing_model.get_holding_cost()
        current_inventory = sourcing_model.get_current_inventory()
        last_cost = holding_cost * torch.relu(
            current_inventory
        ) + shortage_cost * torch.relu(-current_inventory)
        return last_cost

    @no_type_check
    def get_total_cost(
        self,
        sourcing_model: SingleSourcingModel,
        sourcing_periods: int,
        seed: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Calculate the total cost for single-sourcing optimization.

        Parameters
        ----------
        sourcing_model : SingleSourcingModel
            The sourcing model.
        sourcing_periods : int
            Number of sourcing periods.
        seed : int, optional
            Random seed for reproducibility.

        Returns
        -------
        torch.Tensor
            The total cost.
        """
        if seed is not None:
            torch.manual_seed(seed)

        total_cost = torch.tensor(0.0)
        for _ in range(sourcing_periods):
            current_inventory = sourcing_model.get_current_inventory()
            past_orders = sourcing_model.get_past_orders()
            q = torch.as_tensor(
                self.predict(current_inventory, past_orders, output_tensor=True)
            )
            sourcing_model.order(q)
            last_cost = self.get_last_cost(sourcing_model)
            total_cost += last_cost.mean()
        return total_cost

    @no_type_check
    def get_average_cost(
        self,
        sourcing_model: SingleSourcingModel,
        sourcing_periods: int,
        seed: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Calculate the average cost for single-sourcing optimization.

        Parameters
        ----------
        sourcing_model : SourcingModel
            The sourcing model.
        sourcing_periods : int
            Number of sourcing periods.
        seed : int, optional
            Random seed for reproducibility.

        Returns
        -------
        torch.Tensor
            The average cost.
        """
        return (
            self.get_total_cost(sourcing_model, sourcing_periods, seed)
            / sourcing_periods
        )

    @no_type_check
    def simulate(
        self,
        sourcing_model: SingleSourcingModel,
        sourcing_periods: int,
        seed: Optional[int] = None,
    ) -> tuple[NDArray, NDArray]:
        """
        Simulate the sourcing model's output using the given controller.

        Parameters
        ----------
        sourcing_model : SingleSourcingModel
            The sourcing model.
        sourcing_periods : int
            Number of sourcing periods.
        seed : int, optional
            Random seed for reproducibility.

        Returns
        -------
        past_inventories : np.array
            Array of past inventories.
        past_orders : np.array
            Array of past orders.

        """
        if seed is not None:
            torch.manual_seed(seed)
        sourcing_model.reset(batch_size=1)
        for _ in range(sourcing_periods):
            current_inventory = sourcing_model.get_current_inventory()
            past_orders = sourcing_model.get_past_orders()
            q = torch.as_tensor(
                self.predict(current_inventory, past_orders, output_tensor=True)
            )
            sourcing_model.order(q)
        past_inventories = sourcing_model.get_past_inventories()[0, :].detach().numpy()
        past_orders = sourcing_model.get_past_orders()[0, :].detach().numpy()
        return past_inventories, past_orders

    def plot(
        self,
        sourcing_model: SingleSourcingModel,
        sourcing_periods: int,
        linewidth: int = 1,
        seed: Optional[int] = None,
    ) -> tuple[plt.Figure, plt.Axes]:
        """
        Plot the inventory and order quantities over a given number of sourcing periods.

        Parameters
        ----------
        sourcing_model : SingleSourcingModel
            The sourcing model to be used for plotting.
        sourcing_periods : int
            The number of sourcing periods for plotting.
        linewidth : int, default is 1
            The width of the line in the step plots.
        seed : int, optional
            Random seed for reproducibility.

        Returns
        -------
        plt.Figure
            The matplotlib Figure object containing the plots.
        plt.Axes
            The matplotlib Axes objects containing the plots.
        """
        past_inventories, past_orders = self.simulate(
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
            past_orders[-sourcing_periods:],
            linewidth=linewidth,
            color="tab:orange",
        )
        ax[1].yaxis.get_major_locator().set_params(integer=True)
        ax[1].set_title("Order")
        ax[1].set_xlabel("Period")
        ax[1].set_ylabel("Quantity")

        return fig, ax
