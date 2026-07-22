import pytest
import torch

from idinn.demand import CustomDemand
from idinn.sourcing_model import SingleSourcingModel


@pytest.fixture
def single_sourcing_model_custom_demand():
    """
    Single sourcing model with fixed demand at 1.
    """
    return SingleSourcingModel(
        lead_time=1,
        holding_cost=0.5,
        shortage_cost=1,
        init_inventory=10,
        batch_size=3,
        demand_generator=CustomDemand({1: 1.0}),
    )


def test_custom_demand_initialization(
    single_sourcing_model_custom_demand: SingleSourcingModel,
):
    assert (
        single_sourcing_model_custom_demand.get_lead_time() == 1
    ), "Lead time should be 1."
    assert torch.all(
        single_sourcing_model_custom_demand.get_init_inventory() == torch.tensor([10.0])
    ), "Initial inventory should be 10."


def test_custom_demand_order(
    single_sourcing_model_custom_demand: SingleSourcingModel,
):
    # Simulate orders and ensure inventory updates correctly with demand fixed at 1
    single_sourcing_model_custom_demand.order(torch.tensor([[6.0], [6.0], [6.0]]))
    single_sourcing_model_custom_demand.order(torch.tensor([[0.0], [0.0], [0.0]]))
    current_inventory = single_sourcing_model_custom_demand.get_current_inventory()
    expected_inventory = torch.tensor(
        [[14.0], [14.0], [14.0]]
    )  # 10 initial + 6 orders - 2 demand
    assert torch.all(
        current_inventory == expected_inventory
    ), f"Expected inventory {expected_inventory}, but got {current_inventory}."
    past_demands = single_sourcing_model_custom_demand.get_past_demands()
    assert torch.all(
        past_demands[:, -1] == 1
    ), "Custom demand should always generate a demand of 1."


def test_custom_demand_past_orders(
    single_sourcing_model_custom_demand: SingleSourcingModel,
):
    single_sourcing_model_custom_demand.order(torch.tensor([[2.0], [3.0], [4.0]]))
    past_orders = single_sourcing_model_custom_demand.get_past_orders()
    assert torch.all(
        torch.eq(past_orders[:, -1], torch.tensor([2.0, 3.0, 4.0]))
    ), f"Past orders should match the last orders made. Got {past_orders[:, -1]}."


def test_custom_demand_multiple_periods(
    single_sourcing_model_custom_demand: SingleSourcingModel,
):
    for _ in range(5):
        single_sourcing_model_custom_demand.order(
            torch.tensor([[1.0], [1.0], [1.0]]), seed=42
        )
    past_demands = single_sourcing_model_custom_demand.get_past_demands()
    past_inventories = single_sourcing_model_custom_demand.get_past_inventories()
    assert past_demands.shape == (
        3,
        6,
    ), f"Expected demand shape (3, 6), got {past_demands.shape}."
    assert past_inventories.shape == (
        3,
        6,
    ), f"Expected inventory shape (3, 6), got {past_inventories.shape}."
    assert torch.all(
        past_demands[:, -1] == 1
    ), "Custom demand should generate a demand of 1 for all periods."


def test_custom_demand_reset(single_sourcing_model_custom_demand: SingleSourcingModel):
    # Test resetting the model
    single_sourcing_model_custom_demand.order(
        torch.tensor([[2.0], [2.0], [2.0]]), seed=42
    )
    single_sourcing_model_custom_demand.reset()
    assert torch.all(
        single_sourcing_model_custom_demand.get_past_orders() == torch.zeros(3, 1)
    ), "Past orders should be reset to zero."
    assert torch.all(
        single_sourcing_model_custom_demand.get_past_demands() == torch.zeros(3, 1)
    ), "Past demands should be reset to zero."
    assert torch.all(
        single_sourcing_model_custom_demand.get_past_inventories()
        == torch.tensor([10.0]).repeat(3, 1)
    ), "Past inventories should reset to initial inventory."
