from abc import ABCMeta, abstractmethod
from typing import List, Optional, Tuple, Union, no_type_check

import numpy as np
import torch
from matplotlib import pyplot as plt
from numpy.typing import NDArray

from ..sourcing_model import DualSourcingModel


from typing import Optional, Union, List, Tuple
from abc import abstractmethod, ABC
import torch

from ..sourcing_model import DualSourcingModel

class BaseDPController(ABC):

    """
    Base class for DP Controllers 
    """

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

    @abstractmethod
    def get_last_cost(self, sourcing_model: DualSourcingModel) -> torch.Tensor:
        """
        Calculate the cost for the latest period.
        """
        pass 

    @abstractmethod
    def get_total_cost(self, sourcing_model, sorcing_periods, seed) -> torch.torch.Tensor:
        """
        Accumlate cost over sourcing periods 
        """
        pass 

    @abstractmethod
    def get_average_cost(self, sourcing_model, sourcing_periods, seed) -> torch.Tensor:
        """
        Return average cost over sourcing periods
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



class BaseNeuralController(ABC):

    """
    Base class for Neural Network Controllers 
    """

    @abstractmethod
    def init_layers(self) -> None:
        """
        Implement complete NN architecture
        """
        pass

    @abstractmethod
    def prepare_inputs(self) -> torch.Tensor:
        """
        Modify input state to a Tensor, align with the input layer of NN
        """
        pass 

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

    @abstractmethod
    def get_last_cost(self, sourcing_model: DualSourcingModel) -> torch.Tensor:
        """
        Calculate the cost for the latest period.
        """
        pass 

    @abstractmethod
    def get_total_cost(self, sourcing_model, sorcing_periods, seed) -> torch.torch.Tensor:
        """
        Accumlate cost over sourcing periods 
        """
        pass 

    @abstractmethod
    def get_average_cost(self, sourcing_model, sourcing_periods, seed) -> torch.Tensor:
        """
        Return average cost over sourcing periods
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
