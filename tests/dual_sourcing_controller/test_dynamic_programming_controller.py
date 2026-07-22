import pytest

from idinn.demand import UniformDemand
from idinn.dual_controller import DynamicProgrammingController
from idinn.sourcing_model import DualSourcingModel


@pytest.fixture
def dual_sourcing_model_dp():
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
def trained_dp_controller():
    dual_sourcing_model_train = DualSourcingModel(
        regular_lead_time=2,
        expedited_lead_time=0,
        regular_order_cost=0,
        expedited_order_cost=20,
        holding_cost=5,
        shortage_cost=495,
        init_inventory=0,
        demand_generator=UniformDemand(low=0, high=4),
    )
    controller_dp = DynamicProgrammingController()
    controller_dp.fit(dual_sourcing_model_train, max_iterations=101, tolerance=1, validation_freq=100,)
    return controller_dp


def test_dp_controller_avg_cost(dual_sourcing_model_dp, trained_dp_controller):
    # Calculate average cost
    avg_cost = trained_dp_controller.get_average_cost(
        dual_sourcing_model_dp, sourcing_periods=1000, seed=42
    )
    assert abs(avg_cost - 24) < 1, f"Average cost should be near 24, but got {avg_cost}"


def test_dp_controller_plot(dual_sourcing_model_dp, trained_dp_controller):
    # Ensure plotting does not raise errors
    trained_dp_controller.plot(dual_sourcing_model_dp, sourcing_periods=100)


def test_dp_controller_order_prediction(trained_dp_controller):
    # Predict orders using the trained controller
    regular_order, expedited_order = trained_dp_controller.predict(
        current_inventory=3, past_regular_orders=[1], past_expedited_orders=[0]
    )

    # Validate predictions
    assert (
        regular_order == 3
    ), "Predicted regular order should be 3, but got {regular_order}."
    assert (
        expedited_order == 1
    ), "Predicted expedited order should be 1, but got {expedited_order}."

def test_cdi_controller_simulate(dual_sourcing_model_dp, trained_dp_controller):
    # Simulate
    past_inventories, past_regular_orders, past_expedited_orders = (
        trained_dp_controller.simulate(
            dual_sourcing_model_dp, sourcing_periods=100, seed=42
        )
    )

    # Validate simulation results
    assert (
        len(past_inventories) == 101
    ), "Simulation did not return correct number of inventory records."
    assert (
        len(past_regular_orders) == 101
    ), "Simulation did not return correct number of regular order records."
    assert (
        len(past_expedited_orders) == 101
    ), "Simulation did not return correct number of expedited order records."
