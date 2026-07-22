import pytest
import torch

from idinn.demand import CustomDemand
from idinn.sourcing_model import DualSourcingModel


@pytest.fixture
def dual_sourcing_model_custom_demand():
    """
    Dual sourcing model with fixed demand at 1.
    """
    return DualSourcingModel(
        regular_lead_time=1,
        expedited_lead_time=0,
        regular_order_cost=0,
        expedited_order_cost=20,
        holding_cost=0.5,
        shortage_cost=1,
        init_inventory=10,
        batch_size=3,
        demand_generator=CustomDemand({1: 1.0}),
    )


def test_custom_demand_initialization(
    dual_sourcing_model_custom_demand: DualSourcingModel,
):
    # Test that the model initializes correctly
    assert (
        dual_sourcing_model_custom_demand.get_regular_lead_time() == 1
    ), "Regular lead time should be 1."
    assert (
        dual_sourcing_model_custom_demand.get_expedited_lead_time() == 0
    ), "Expedited lead time should be 0."
    assert torch.all(
        dual_sourcing_model_custom_demand.get_init_inventory() == torch.tensor([10.0])
    ), "Initial inventory should be 10."


def test_custom_demand_order(
    dual_sourcing_model_custom_demand: DualSourcingModel,
):
    # Test that the demand generator consistently outputs 1.0
    dual_sourcing_model_custom_demand.order(
        torch.tensor([[3.0], [3.0], [3.0]]), torch.tensor([[1.0], [1.0], [1.0]])
    )
    past_demands = dual_sourcing_model_custom_demand.get_past_demands()
    assert torch.all(
        past_demands[:, -1] == 1.0
    ), "Custom demand should consistently output a fixed value of 1.0."

    # Test that inventories update correctly with regular and expedited orders
    dual_sourcing_model_custom_demand.order(
        torch.tensor([[1.0], [1.0], [1.0]]), torch.tensor([[1.0], [1.0], [1.0]])
    )
    current_inventory = dual_sourcing_model_custom_demand.get_current_inventory()
    expected_inventory = torch.tensor(
        [[13.0], [13.0], [13.0]]
    )  # 10 initial + 3 regular + 2 expedited - 2 demand
    assert torch.all(
        current_inventory == expected_inventory
    ), f"Expected inventory {expected_inventory}, got {current_inventory}."


def test_custom_demand_past_orders(
    dual_sourcing_model_custom_demand: DualSourcingModel,
):
    # Test that both regular and expedited past orders are tracked correctly
    regular_orders = torch.tensor([[5.0], [6.0], [7.0]])
    expedited_orders = torch.tensor([[1.0], [1.0], [1.0]])
    dual_sourcing_model_custom_demand.order(regular_orders, expedited_orders)
    past_regular_orders = dual_sourcing_model_custom_demand.get_past_regular_orders()
    past_expedited_orders = (
        dual_sourcing_model_custom_demand.get_past_expedited_orders()
    )
    assert torch.all(
        past_regular_orders[:, -1] == regular_orders.squeeze()
    ), "Last regular orders do not match expected values."
    assert torch.all(
        past_expedited_orders[:, -1] == expedited_orders.squeeze()
    ), "Last expedited orders do not match expected values."


def test_custom_demand_multiple_periods(
    dual_sourcing_model_custom_demand: DualSourcingModel,
):
    # Test model behavior over multiple periods
    for _ in range(5):
        dual_sourcing_model_custom_demand.order(
            torch.tensor([[2.0], [2.0], [2.0]]), torch.tensor([[1.0], [1.0], [1.0]])
        )
    past_inventories = dual_sourcing_model_custom_demand.get_past_inventories()
    past_demands = dual_sourcing_model_custom_demand.get_past_demands()
    assert past_inventories.shape == (
        3,
        6,
    ), f"Expected inventory shape (3, 6), got {past_inventories.shape}."
    assert past_demands.shape == (
        3,
        6,
    ), f"Expected demand shape (3, 6), got {past_demands.shape}."
    assert torch.all(
        past_demands[:, -1] == 1.0
    ), "Custom demand should consistently output a fixed value of 1.0."


def test_custom_demand_reset(dual_sourcing_model_custom_demand: DualSourcingModel):
    # Test resetting the model
    dual_sourcing_model_custom_demand.order(
        torch.tensor([[2.0], [2.0], [2.0]]), torch.tensor([[1.0], [1.0], [1.0]])
    )
    dual_sourcing_model_custom_demand.reset()
    assert torch.all(
        dual_sourcing_model_custom_demand.get_past_regular_orders() == torch.zeros(3, 1)
    ), "Past regular orders should be reset to zero."
    assert torch.all(
        dual_sourcing_model_custom_demand.get_past_expedited_orders()
        == torch.zeros(3, 1)
    ), "Past expedited orders should be reset to zero."
    assert torch.all(
        dual_sourcing_model_custom_demand.get_past_demands() == torch.zeros(3, 1)
    ), "Past demands should be reset to zero."
    assert torch.all(
        dual_sourcing_model_custom_demand.get_past_inventories()
        == torch.tensor([10.0]).repeat(3, 1)
    ), "Past inventories should reset to initial inventory."
