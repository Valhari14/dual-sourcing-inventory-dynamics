import pytest
import torch

from idinn.demand import UniformDemand
from idinn.dual_controller import CappedDualIndexController
from idinn.sourcing_model import DualSourcingModel


@pytest.fixture
def dual_sourcing_model_cdi():
    return DualSourcingModel(
        regular_lead_time=2,
        expedited_lead_time=0,
        regular_order_cost=0,
        expedited_order_cost=20,
        holding_cost=5,
        shortage_cost=495,
        init_inventory=0,
        demand_generator=UniformDemand(low=0, high=4),
    )


@pytest.fixture(scope="module")
def trained_cdi_controller():
    sourcing_model_train = DualSourcingModel(
        regular_lead_time=2,
        expedited_lead_time=0,
        regular_order_cost=0,
        expedited_order_cost=20,
        holding_cost=5,
        shortage_cost=495,
        init_inventory=0,
        demand_generator=UniformDemand(low=0, high=4),
    )
    controller_cdi = CappedDualIndexController()
    controller_cdi.fit(
        sourcing_model_train,
        sourcing_periods=100,
        s_e_range=torch.arange(4, 5),
        s_r_range=torch.arange(8, 9),
        q_r_range=torch.arange(2, 3),
    )
    return controller_cdi


def test_cdi_controller_avg_cost(dual_sourcing_model_cdi, trained_cdi_controller):
    # Calculate average cost
    avg_cost = trained_cdi_controller.get_average_cost(
        dual_sourcing_model_cdi, sourcing_periods=1000, seed=42
    )
    assert abs(avg_cost - 26) < 1, f"Average cost should be near 26, but got {avg_cost}"


def test_cdi_controller_simulate(dual_sourcing_model_cdi, trained_cdi_controller):
    # Simulate
    past_inventories, past_regular_orders, past_expedited_orders = (
        trained_cdi_controller.simulate(
            dual_sourcing_model_cdi, sourcing_periods=100, seed=42
        )
    )

    # Validate simulation results
    assert len(past_inventories) == 101, (
        "Simulation did not return correct number of inventory records."
    )
    assert len(past_regular_orders) == 101, (
        "Simulation did not return correct number of regular order records."
    )
    assert len(past_expedited_orders) == 101, (
        "Simulation did not return correct number of expedited order records."
    )


def test_cdi_controller_plot(dual_sourcing_model_cdi, trained_cdi_controller):
    # Ensure plotting does not raise errors
    trained_cdi_controller.plot(dual_sourcing_model_cdi, sourcing_periods=100, seed=42)


def test_cdi_controller_order_prediction(trained_cdi_controller):
    # Predict orders using the trained controller
    regular_order, expedited_order = trained_cdi_controller.predict(
        current_inventory=3,
        past_regular_orders=[1],
        past_expedited_orders=[0],
    )

    # Validate predictions
    assert regular_order == 2, (
        f"Predicted regular order should be 2, but got {regular_order}."
    )
    assert expedited_order == 1, (
        f"Predicted expedited order should be 1, but got {expedited_order}."
    )
