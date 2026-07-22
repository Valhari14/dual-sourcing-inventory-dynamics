import pytest
import torch

from idinn.demand import UniformDemand
from idinn.sourcing_model import DualSourcingModel


@pytest.fixture
def dual_sourcing_model():
    return DualSourcingModel(
        regular_lead_time=2,
        expedited_lead_time=1,
        regular_order_cost=0,
        expedited_order_cost=20,
        holding_cost=0.5,
        shortage_cost=1,
        init_inventory=10,
        demand_generator=UniformDemand(low=1, high=4),
        batch_size=3,
    )


def test_dual_sourcing_model_initialization(dual_sourcing_model: DualSourcingModel):
    assert (
        dual_sourcing_model.get_regular_lead_time() == 2
    ), "Regular lead time should be 2."
    assert (
        dual_sourcing_model.get_expedited_lead_time() == 1
    ), "Expedited lead time should be 1."
    assert (
        dual_sourcing_model.get_regular_order_cost() == 0
    ), "Regular order cost should be 0."
    assert (
        dual_sourcing_model.get_expedited_order_cost() == 20
    ), "Expedited order cost should be 20."
    assert torch.all(
        dual_sourcing_model.get_init_inventory() == torch.tensor([10.0])
    ), "Initial inventory should be 10."


def test_dual_sourcing_model_order(dual_sourcing_model: DualSourcingModel):
    # Test order function for both regular and expedited orders
    regular_orders = torch.tensor([[5.0], [6.0], [7.0]])
    expedited_orders = torch.tensor([[2.0], [2.0], [2.0]])
    dual_sourcing_model.order(regular_orders, expedited_orders, seed=42)
    assert dual_sourcing_model.get_past_regular_orders().shape == (
        3,
        2,
    ), "Regular orders should be updated correctly."
    assert dual_sourcing_model.get_past_expedited_orders().shape == (
        3,
        2,
    ), "Expedited orders should be updated correctly."
    assert torch.all(
        torch.eq(dual_sourcing_model.get_last_regular_order(), regular_orders)
    ), "Last regular orders do not match expected values."
    assert torch.all(
        torch.eq(dual_sourcing_model.get_last_expedited_order(), expedited_orders)
    ), "Last expedited orders do not match expected values."

    # Test inventory updates with regular and expedited orders
    current_inventory = dual_sourcing_model.get_current_inventory()
    expected_inventory = torch.tensor([[6.0], [6.0], [8.0]])
    assert torch.all(
        current_inventory == expected_inventory
    ), f"Expected inventory {expected_inventory}, got {current_inventory}."


def test_dual_sourcing_model_multiple_periods(dual_sourcing_model: DualSourcingModel):
    # Test demand tracking across multiple periods
    for _ in range(3):
        dual_sourcing_model.order(
            torch.tensor([[1.0], [1.0], [1.0]]),
            torch.tensor([[1.0], [1.0], [1.0]]),
            seed=42,
        )
    past_demands = dual_sourcing_model.get_past_demands()
    assert past_demands.shape == (
        3,
        4,
    ), f"Expected demand shape (3, 4), got {past_demands.shape}."
    assert torch.all(past_demands[:, -1] >= 1) and torch.all(
        past_demands[:, -1] <= 4
    ), "Demands should respect the generator's range."


def test_dual_sourcing_model_reset(dual_sourcing_model: DualSourcingModel):
    # Test the reset function
    dual_sourcing_model.order(
        torch.tensor([[2.0], [3.0], [4.0]]), torch.tensor([[1.0], [1.0], [1.0]])
    )
    dual_sourcing_model.reset(batch_size=5)
    assert (
        dual_sourcing_model.batch_size == 5
    ), "Batch size should be updated to 5 after reset."
    assert dual_sourcing_model.get_past_regular_orders().shape == (
        5,
        1,
    ), "Past regular orders should be reset."
    assert dual_sourcing_model.get_past_expedited_orders().shape == (
        5,
        1,
    ), "Past expedited orders should be reset."
    assert dual_sourcing_model.get_past_inventories().shape == (
        5,
        1,
    ), "Past inventories should be reset to initial values."
