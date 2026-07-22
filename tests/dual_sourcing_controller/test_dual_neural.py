import pytest
import torch

from idinn.demand import UniformDemand
from idinn.dual_controller import DualSourcingNeuralController
from idinn.sourcing_model import DualSourcingModel

torch.set_default_device('cpu')
# TODO: CUDA is numerically unstable,
#  may need new tests!


@pytest.fixture
def dual_sourcing_model_neural():
    return DualSourcingModel(
        regular_lead_time=2,
        expedited_lead_time=0,
        regular_order_cost=0,
        expedited_order_cost=20,
        holding_cost=5,
        shortage_cost=495,
        batch_size=256,
        init_inventory=6,
        demand_generator=UniformDemand(low=0, high=4),
    )


@pytest.fixture(scope="module")
def trained_neural_controller():
    dual_sourcing_model_train = DualSourcingModel(
        regular_lead_time=2,
        expedited_lead_time=0,
        regular_order_cost=0,
        expedited_order_cost=20,
        holding_cost=5,
        shortage_cost=495,
        batch_size=8,
        init_inventory=6,
        demand_generator=UniformDemand(low=0, high=4),
    )
    controller_neural = DualSourcingNeuralController(
        hidden_layers=[128, 64, 32, 16, 8, 4],
        activation=torch.nn.CELU(alpha=1),
    )
    controller_neural.fit(
        sourcing_model=dual_sourcing_model_train,
        sourcing_periods=100,
        validation_sourcing_periods=1,
        epochs=2,
        seed=1234,
    )
    return controller_neural


def test_neural_controller_init_layers(
    trained_neural_controller: DualSourcingNeuralController,
):
    assert (
        len(trained_neural_controller.model) == 14
    ), f"Model should have 14 layers, but got {len(trained_neural_controller.model)}"


def test_neural_controller_avg_cost(
    dual_sourcing_model_neural, trained_neural_controller
):
    # Calculate average cost
    avg_cost = trained_neural_controller.get_average_cost(
        dual_sourcing_model_neural, sourcing_periods=1000
    )
    assert (
        abs(avg_cost - 51133.42) < 1
    ), f"Average cost should be near 51133.42, but got {avg_cost}"


def test_neural_controller_simulate(
    dual_sourcing_model_neural, trained_neural_controller
):
    # TODO: Simulate is not working properly
    past_inventories, past_regular_orders, past_expedited_orders = (
        trained_neural_controller.simulate(
            dual_sourcing_model_neural, sourcing_periods=100
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


def test_neural_controller_plot(dual_sourcing_model_neural, trained_neural_controller):
    # Ensure plotting does not raise errors
    trained_neural_controller.plot(dual_sourcing_model_neural, sourcing_periods=100)


def test_neural_controller_order_prediction(
    dual_sourcing_model_neural, trained_neural_controller
):
    # Predict orders using the trained controller
    regular_order, expedited_order = trained_neural_controller.predict(
        current_inventory=3, past_regular_orders=[1], past_expedited_orders=[0]
    )

    # Validate predictions
    assert (
        regular_order == 0
    ), "Predicted regular order should be 0, but got {regular_order}."
    assert (
        expedited_order == 0
    ), "Predicted expedited order should be 0, but got {expedited_order}."
