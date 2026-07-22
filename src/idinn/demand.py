from abc import ABCMeta, abstractmethod

import torch


class BaseDemand(metaclass=ABCMeta):
    @abstractmethod
    def sample(self, batch_size: int) -> torch.Tensor:
        """
        Generate demand for one period.

        Parameters
        ----------
        batch_size: int
            Size of generated demands which should correspond to the batch size or the number of SKUs.
        """
        pass

    @abstractmethod
    def enumerate_support(self) -> dict:
        pass

    @abstractmethod
    def get_min_demand(self) -> int:
        pass

    @abstractmethod
    def get_max_demand(self) -> int:
        pass


class UniformDemand(BaseDemand):
    def __init__(self, low: int, high: int):
        self.distribution = torch.distributions.Uniform(low=low, high=high + 1)
        self.demand_prob = 1 / (high - low + 1)
        self.min_demand = low
        self.max_demand = high

    def sample(self, batch_size: int, batch_width: int = 1) -> torch.Tensor:
        return self.distribution.sample([batch_size, batch_width]).int()

    def enumerate_support(self) -> dict:
        return {
            x: 1 / (self.max_demand + 1 - self.min_demand)
            for x in range(self.min_demand, self.max_demand + 1)
        }

    def get_min_demand(self) -> int:
        return self.min_demand

    def get_max_demand(self) -> int:
        return self.max_demand


class CustomDemand(BaseDemand):
    def __init__(self, demand_prob: dict[int, float]):
        from math import isclose

        # All demand values should be int
        for key in demand_prob:
            if not isinstance(key, int):
                raise TypeError(f"Demand values '{key}' is not an integer.")
        # All demand probabilities should be float
        for value in demand_prob.values():
            if not isinstance(value, (int, float)):
                raise TypeError(f"Demand probabilities '{value}' is not a float.")
        # Sum of probabilities should be close to 1
        total = sum(demand_prob.values())
        if not isclose(total, 1, abs_tol=1e-3):
            raise ValueError(
                f"The sum of demand probablities is {total}, which is not close to 1."
            )

        self.demand_prob = demand_prob

    def sample(self, batch_size: int, batch_width: int = 1) -> torch.Tensor:
        """
        Generate demand for one period.

        Parameters
        ----------
        batch_size: int
            Size of generated demands which should correspond to the batch size or the number of SKUs. If the size does not match the dimension of the elements from `demand_history`, demand will be upsampled or downsampled to match the size.
        """
        # Require Python >= 3.8 thus order is preserved (matches insertion order)
        # Draw dictionary keys with corresponding probabilities
        sampled_indices = torch.multinomial(
            torch.tensor(list(self.demand_prob.values())),
            num_samples=batch_size * batch_width,
            replacement=True,
        )
        return torch.tensor(list(self.demand_prob.keys()))[sampled_indices].reshape(
            batch_size, batch_width
        )

    def enumerate_support(self) -> dict:
        return self.demand_prob

    def get_min_demand(self) -> int:
        return min(self.demand_prob.keys())

    def get_max_demand(self) -> int:
        return max(self.demand_prob.keys())
