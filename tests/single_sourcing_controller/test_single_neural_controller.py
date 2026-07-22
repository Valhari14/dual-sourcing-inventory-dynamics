import pytest
import torch

from idinn.demand import UniformDemand
from idinn.single_controller import SingleSourcingNeuralController
from idinn.sourcing_model import SingleSourcingModel


@pytest.fixture
def single_sourcing_model_neural():
    return SingleSourcingModel(
        lead_time=0,
        holding_cost=5,
        shortage_cost=495,
        batch_size=32,
        init_inventory=10,
        demand_generator=UniformDemand(low=1, high=4),
    )


@pytest.fixture(scope="module")
def trained_neural_controller():
    sourcing_model_train = SingleSourcingModel(
        lead_time=0,
        holding_cost=5,
        shortage_cost=495,
        batch_size=4,
        init_inventory=10,
        demand_generator=UniformDemand(low=1, high=4),
    )
    neural_controller = SingleSourcingNeuralController(
        hidden_layers=[2], activation=torch.nn.CELU(alpha=1)
    )
    neural_controller.fit(
        sourcing_model=sourcing_model_train,
        sourcing_periods=5,
        validation_sourcing_periods=1,
        epochs=2,
        seed=1,
    )
    return neural_controller


def test_neural_controller_init_layers(
    trained_neural_controller: SingleSourcingNeuralController,
):
    assert (
        len(trained_neural_controller.model) == 4
    ), f"Model should have 4 layers, but got {len(trained_neural_controller.model)}"


def test_neural_controller_avg_cost(
    single_sourcing_model_neural, trained_neural_controller
):
    # Calculate average cost
    avg_cost = trained_neural_controller.get_average_cost(
        single_sourcing_model_neural, sourcing_periods=1000, seed=42
    )
    assert (
        abs(avg_cost - 4681.83) < 1
    ), f"Average cost should be near 4681.83 but got {avg_cost}"


def test_neural_controller_simulate(
    single_sourcing_model_neural, trained_neural_controller
):
    # Simulate sourcing
    past_inventories, past_orders = trained_neural_controller.simulate(
        single_sourcing_model_neural, sourcing_periods=100
    )

    # Validate simulation results
    assert (
        len(past_inventories) == 101
    ), "Simulation did not return correct number of inventory records."
    assert (
        len(past_orders) == 101
    ), "Simulation did not return correct number of order records."


def test_neural_controller_plot(
    single_sourcing_model_neural, trained_neural_controller
):
    # Ensure plotting does not raise errors
    trained_neural_controller.plot(single_sourcing_model_neural, sourcing_periods=100)


def test_neural_controller_predict(trained_neural_controller):
    # Predict order quantity
    predicted_order = trained_neural_controller.predict(current_inventory=0)

    # Validate prediction
    assert predicted_order is not None, "Predicted order should not be None."
    assert int(predicted_order) == 0, "Predicted order quantity should be 4."
