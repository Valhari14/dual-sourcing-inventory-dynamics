Save and Load Controllers
=========================

Trained controllers can be saved to disk for later use. For the :class:`SingleSourcingNeuralController` and :class:`DualSourcingNeuralController`, this can be achieved using their `save` and `load` methods.

.. doctest::
    >>> from idinn.single_controller import SingleSourcingNeuralController

    >>> # Save the model
    >>> single_neural_controller.save("optimal_single_neural_controller.pt")
    >>> # Load the model
    >>> saved_single_controller = SingleSourcingNeuralController()
    >>> saved_single_controller = saved_single_controller.load("optimal_single_neural_controller.pt")

For other controllers, Python's pickle utility can be used instead.

.. doctest::

    >>> import pickle
    >>> from idinn.sourcing_model import SingleSourcingModel
    >>> from idinn.single_controller import BaseStockController
    >>> from idinn.demand import UniformDemand
    
    >>> single_sourcing_model = SingleSourcingModel(
    ...     lead_time=2,
    ...     holding_cost=5,
    ...     shortage_cost=495,
    ...     batch_size=32,
    ...     init_inventory=10,
    ...     demand_generator=UniformDemand(low=0, high=4),
    ... )
    >>> controller_base = BaseStockController()
    >>> controller_base.fit(single_sourcing_model)

    # Save trained controller to "controller_base.pkl"
    >>> with open("controller_base.pkl", "wb") as f:
    ...     pickle.dump(controller_base, f)

    # The file can later be loaded
    >>> with open("controller_base.pkl", "rb") as f:
    ...     controller_saved = pickle.load(f)