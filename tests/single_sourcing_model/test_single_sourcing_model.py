import pytest
import torch

from idinn.demand import UniformDemand
from idinn.sourcing_model import SingleSourcingModel


@pytest.fixture
def single_sourcing_model():
    return SingleSourcingModel(
        lead_time=2,
        holding_cost=0.5,
        shortage_cost=1,
        init_inventory=10,
        demand_generator=UniformDemand(low=1, high=4),
        batch_size=3,
    )


def test_single_sourcing_model_initialization(
    single_sourcing_model: SingleSourcingModel,
):
    assert single_sourcing_model.get_lead_time() == 2, "Lead time should be 2."
    assert single_sourcing_model.get_holding_cost() == 0.5, (
        "Holding cost should be 0.5."
    )
    assert torch.all(
        torch.eq(single_sourcing_model.get_init_inventory(), torch.tensor([10.0]))
    ), "Inital inventory should be 10."


def test_single_sourcing_model_order(single_sourcing_model: SingleSourcingModel):
    q = torch.tensor([[5.0], [10.0], [15.0]])
    single_sourcing_model.order(q)
    past_orders = single_sourcing_model.get_past_orders()
    assert past_orders.shape == (
        3,
        2,
    ), f"Past orders tensor should have shape (3, 2), got {past_orders.shape}."
    assert torch.all(torch.eq(past_orders[:, -1], q.squeeze())), (
        f"Last order should match {q.squeeze()}."
    )


def test_single_sourcing_model_update_inventories(
    single_sourcing_model: SingleSourcingModel,
):
    q = torch.tensor([[5.0], [10.0], [15.0]])
    single_sourcing_model.order(q, seed=42)
    past_inventories = single_sourcing_model.get_past_inventories()
    assert past_inventories.shape == (3, 2), (
        f"Past inventories tensor should have shape (3, 2), got {past_inventories.shape}."
    )
    current_inventory = single_sourcing_model.get_current_inventory()
    assert current_inventory.shape == (3, 1), (
        f"Current inventory tensor should have shape (3, 1), got {current_inventory.shape}."
    )


def test_single_sourcing_model_demands(single_sourcing_model: SingleSourcingModel):
    single_sourcing_model.order(torch.tensor([[10.0], [20.0], [30.0]]), seed=123)
    past_demands = single_sourcing_model.get_past_demands()
    assert past_demands.shape == (
        3,
        2,
    ), f"Past demands tensor should have shape (3, 2), got {past_demands.shape}."
    assert torch.all(past_demands[:, -1] >= 1) and torch.all(
        past_demands[:, -1] <= 4
    ), "Generated demands should be between 1 and 4."


def test_single_sourcing_model_reset(single_sourcing_model: SingleSourcingModel):
    single_sourcing_model.reset(batch_size=5)
    assert single_sourcing_model.batch_size == 5, (
        "Batch size should be updated to 5 after reset."
    )
    assert single_sourcing_model.get_past_orders().shape == (
        5,
        1,
    ), "Past orders tensor should have shape (5, 1)."
    assert single_sourcing_model.get_past_inventories().shape == (
        5,
        1,
    ), "Past inventories tensor should have shape (5, 1)."
