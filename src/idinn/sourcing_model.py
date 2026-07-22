from typing import Optional, Union

import torch

from .demand import CustomDemand, UniformDemand


class BaseSourcingModel:
    def __init__(
        self,
        holding_cost: float,
        shortage_cost: float,
        init_inventory: float,
        demand_generator: Union[UniformDemand, CustomDemand],
        batch_size: int,
        lead_time: Optional[int] = None,
        regular_lead_time: Optional[int] = None,
        expedited_lead_time: Optional[int] = None,
        regular_order_cost: Optional[float] = None,
        expedited_order_cost: Optional[float] = None,
    ):
        self.holding_cost = holding_cost
        self.shortage_cost = shortage_cost
        self.init_inventory = torch.tensor(
            [init_inventory], requires_grad=True, dtype=torch.float
        )
        self.batch_size = batch_size
        self.lead_time = lead_time
        self.regular_lead_time = regular_lead_time
        self.expedited_lead_time = expedited_lead_time
        self.regular_order_cost = regular_order_cost
        self.expedited_order_cost = expedited_order_cost
        self.demand_generator = demand_generator
        self.reset()

    def reset(self, batch_size: Optional[int] = None) -> None:
        if batch_size is not None and self.batch_size != batch_size:
            self.batch_size = batch_size

        self.past_inventories = self.get_init_inventory().repeat(self.batch_size, 1)
        self.past_demands = torch.zeros(self.batch_size, 1)

        if self.lead_time is not None:
            self.past_orders = torch.zeros(self.batch_size, 1)

        elif (
            self.regular_lead_time is not None and self.expedited_lead_time is not None
        ):
            if self.regular_lead_time < self.expedited_lead_time:
                raise ValueError(
                    "`regular_lead_time` must be greater than or equal to expedited_lead_time."
                )
            self.past_regular_orders = torch.zeros(self.batch_size, 1)
            self.past_expedited_orders = torch.zeros(self.batch_size, 1)
        else:
            raise ValueError(
                "Either `lead_time` or (`regular_lead_time` and `expedited_lead_time`) must be provided."
            )

    def get_holding_cost(self) -> float:
        return self.holding_cost

    def get_shortage_cost(self) -> float:
        return self.shortage_cost

    def get_init_inventory(self) -> torch.Tensor:
        init_inventory = (
            self.init_inventory - torch.frac(self.init_inventory).clone().detach()
        )
        return init_inventory

    def get_past_inventories(self) -> torch.Tensor:
        return self.past_inventories

    def get_past_demands(self) -> torch.Tensor:
        return self.past_demands

    def get_current_inventory(self) -> torch.Tensor:
        # Keep dim when slicing past_inventories
        # https://stackoverflow.com/questions/57237352/why-does-torch-slice-lose-dimension
        # https://discuss.pytorch.org/t/solved-simple-question-about-keep-dim-when-slicing-the-tensor/9280
        return self.past_inventories[:, [-1]]


class SingleSourcingModel(BaseSourcingModel):
    def __init__(
        self,
        lead_time: int,
        holding_cost: float,
        shortage_cost: float,
        init_inventory: float,
        demand_generator: Union[UniformDemand, CustomDemand],
        batch_size: int = 1,
    ):
        """
        Parameters
        ----------
        lead_time : int
            The lead time for orders.
        holding_cost : float
            The cost of holding inventory.
        shortage_cost : float
            The cost of inventory shortage.
        init_inventory : float
            The initial inventory.
        demand_generator : Demand
        batch_size : int, optional
            The batch size for orders. default is 1.
        """
        super().__init__(
            lead_time=lead_time,
            holding_cost=holding_cost,
            shortage_cost=shortage_cost,
            init_inventory=init_inventory,
            batch_size=batch_size,
            demand_generator=demand_generator,
        )

    def get_lead_time(self) -> int:
        if self.lead_time is None:
            raise ValueError("`lead_time` is not set")

        return self.lead_time

    def get_past_orders(self) -> torch.Tensor:
        return self.past_orders

    def order(self, q: torch.Tensor, seed: Optional[int] = None) -> None:
        """
        Orders items to the inventory and update the inventory with generated demands.

        Parameters
        ----------
        q : torch.Tensor
            The quantity of items to order.
        seed : int, optional
            Random seed for reproducibility.
        """
        if seed is not None:
            torch.manual_seed(seed)

        if self.lead_time is None:
            raise ValueError("`lead_time` is not set")

        # Current orders are added to past_orders
        self.past_orders = torch.cat([self.past_orders, q], dim=1)
        # Past orders arrived, if past orders are not available, then arrived order is 0
        if self.past_orders.shape[1] >= 1 + self.lead_time:
            arrived_order = self.past_orders[:, [-1 - self.lead_time]]
        else:
            arrived_order = torch.zeros(self.batch_size, 1)
        # Generate current demand
        current_demand = self.demand_generator.sample(self.batch_size)
        # Current demand are added to past_demands
        self.past_demands = torch.cat([self.past_demands, current_demand], dim=1)
        # Update inventory
        current_inventory = (
            self.get_current_inventory() + arrived_order - current_demand
        )
        # Current inventories are added to past_inventories
        self.past_inventories = torch.cat(
            [self.past_inventories, current_inventory], dim=1
        )


class DualSourcingModel(BaseSourcingModel):
    def __init__(
        self,
        regular_lead_time: int,
        expedited_lead_time: int,
        regular_order_cost: float,
        expedited_order_cost: float,
        holding_cost: float,
        shortage_cost: float,
        init_inventory: float,
        demand_generator: Union[UniformDemand, CustomDemand],
        batch_size: int = 1,
    ):
        """
        Parameters
        ----------
        regular_lead_time : int
            The lead time for regular orders.
        expedited_lead_time : int
            The lead time for expedited orders.
        regular_order_cost : float
            The cost of placing a regular order.
        expedited_order_cost : float
            The cost of placing an expedited order.
        holding_cost : float
            The cost of holding inventory.
        shortage_cost : float
            The cost of shortage.
        init_inventory : float
            The initial inventory.
        batch_size : int, default is 1
            The batch size for orders.
        demand : Iterable, torch.Tensor, np.array, or pd.Series, optional
            The array for outputting values of demands.
        demand_distribution : str, default is `uniform`.
            Distribution for generated demand when `demand` is not specified.
        demand_low : int, default is 1.
            Lower bound for generated demand when `demand` is not specified.
        demand_high : int, default is 4.
            Higher bound for generated demand when `demand` is not specified.
        """
        super().__init__(
            regular_lead_time=regular_lead_time,
            expedited_lead_time=expedited_lead_time,
            regular_order_cost=regular_order_cost,
            expedited_order_cost=expedited_order_cost,
            holding_cost=holding_cost,
            shortage_cost=shortage_cost,
            init_inventory=init_inventory,
            batch_size=batch_size,
            demand_generator=demand_generator,
        )

    def get_past_regular_orders(self) -> torch.Tensor:
        return self.past_regular_orders

    def get_past_expedited_orders(self) -> torch.Tensor:
        return self.past_expedited_orders

    def get_last_regular_order(self) -> torch.Tensor:
        return self.past_regular_orders[:, [-1]]

    def get_last_expedited_order(self) -> torch.Tensor:
        return self.past_expedited_orders[:, [-1]]

    def get_regular_lead_time(self) -> int:
        if self.regular_lead_time is None:
            raise ValueError("`regular_lead_time` is not set")

        return self.regular_lead_time

    def get_expedited_lead_time(self) -> int:
        if self.expedited_lead_time is None:
            raise ValueError("`expedited_lead_time` is not set")

        return self.expedited_lead_time

    def get_regular_order_cost(self) -> float:
        if self.regular_order_cost is None:
            raise ValueError("Regular order cost is not set")

        return self.regular_order_cost

    def get_expedited_order_cost(self) -> float:
        if self.expedited_order_cost is None:
            raise ValueError("Expedited order cost is not set")

        return self.expedited_order_cost

    def order(
        self,
        regular_q: torch.Tensor,
        expedited_q: torch.Tensor,
        seed: Optional[int] = None,
    ) -> None:
        """
        Orders items to the inventory and update the inventory with generated demands.

        Parameters
        ----------
        regular_q : torch.Tensor
            The quantity of items to order from the regular supplier.
        expedited_q : torch.Tensor
            The quantity of items to order from the expedited supplier.
        seed : int, optional
            Random seed for reproducibility.
        """
        if seed is not None:
            torch.manual_seed(seed)

        if self.regular_lead_time is None:
            raise ValueError("`regular_lead_time` is not set")
        if self.expedited_lead_time is None:
            raise ValueError("`expedited_lead_time` is not set")

        if not isinstance(regular_q, torch.Tensor):
            regular_q = torch.tensor([[regular_q]])
        if not isinstance(expedited_q, torch.Tensor):
            expedited_q = torch.tensor([[expedited_q]])

        # Current regular order are added to past_regular_orders
        self.past_regular_orders = torch.cat(
            [self.past_regular_orders, regular_q], dim=1
        )
        # Current expedited order are added to past_expedited_orders
        self.past_expedited_orders = torch.cat(
            [self.past_expedited_orders, expedited_q], dim=1
        )
        # Past regular orders arrived,
        # if past regular orders are not available, then arrived order is 0
        if self.past_regular_orders.shape[1] >= 1 + self.regular_lead_time:
            arrived_regular_orders = self.past_regular_orders[
                :, [-1 - self.regular_lead_time]
            ]
        else:
            arrived_regular_orders = torch.zeros(self.batch_size, 1)
        # Past expedited orders arrived,
        # if past expedited orders are not available, then arrived order is 0
        if self.past_expedited_orders.shape[1] >= 1 + self.expedited_lead_time:
            arrived_expedited_orders = self.past_expedited_orders[
                :, [-1 - self.expedited_lead_time]
            ]
        else:
            arrived_expedited_orders = torch.zeros(self.batch_size, 1)
        # Generate current demand
        current_demand = self.demand_generator.sample(self.batch_size)
        # Current demand are added to past_demands
        self.past_demands = torch.cat([self.past_demands, current_demand], dim=1)
        # Update inventory
        current_inventory = (
            self.get_current_inventory()
            + arrived_expedited_orders
            + arrived_regular_orders
            - current_demand
        )
        self.past_inventories = torch.cat(
            [self.past_inventories, current_inventory], dim=1
        )
