import pytest

from idinn.demand import UniformDemand
from idinn.single_controller import BaseStockController
from idinn.sourcing_model import SingleSourcingModel


@pytest.fixture
def base_stock_controller():
    return BaseStockController()


@pytest.mark.parametrize(
    "lead_time, expected_z_star, expected_avg_cost",
    [
        (0, 4, 10),
        (2, 11, 29),
    ],
)
def test_base_stock_controller_fit_and_cost(
    base_stock_controller, lead_time, expected_z_star, expected_avg_cost
):
    single_sourcing_model = SingleSourcingModel(
        lead_time=lead_time,
        holding_cost=5,
        shortage_cost=495,
        batch_size=32,
        init_inventory=10,
        demand_generator=UniformDemand(low=0, high=4),
    )
    base_stock_controller.fit(single_sourcing_model, seed=42)
    assert (
        base_stock_controller.z_star == expected_z_star
    ), f"Expected z_star to be {expected_z_star}, but got {base_stock_controller.z_star}"

    avg_cost = base_stock_controller.get_average_cost(
        single_sourcing_model, sourcing_periods=1000, seed=42
    )
    assert (
        abs(avg_cost - expected_avg_cost) < 1
    ), f"Average cost should be near {expected_avg_cost}, but got {avg_cost}"


def test_base_stock_controller_simulation(base_stock_controller):
    single_sourcing_model = SingleSourcingModel(
        lead_time=0,
        holding_cost=5,
        shortage_cost=495,
        batch_size=32,
        init_inventory=10,
        demand_generator=UniformDemand(low=0, high=4),
    )
    base_stock_controller.fit(single_sourcing_model, seed=42)

    past_inventories, past_orders = base_stock_controller.simulate(
        single_sourcing_model, sourcing_periods=100
    )
    assert (
        len(past_inventories) == 101
    ), "Simulation did not return correct inventory records"
    assert len(past_orders) == 101, "Simulation did not return correct order records"


@pytest.mark.parametrize(
    "lead_time, current_inventory, expected_predict",
    [
        (0, 10, 0),
        (2, 10, 1),
    ],
)
def test_base_stock_controller_predict(
    lead_time, current_inventory, expected_predict, base_stock_controller
):
    single_sourcing_model = SingleSourcingModel(
        lead_time=lead_time,
        holding_cost=5,
        shortage_cost=495,
        batch_size=32,
        init_inventory=10,
        demand_generator=UniformDemand(low=0, high=4),
    )
    base_stock_controller.fit(single_sourcing_model, seed=42)
    predict = base_stock_controller.predict(current_inventory=current_inventory)
    assert (
        predict == expected_predict
    ), f"Expected predict to be {expected_predict}, but got {predict}"
