Log with Tensorboard
====================

To better monitor the training process for neural controllers, i.e. :class:`SingleSourcingNeuralController` and :class:`DualSourcingNeuralController`, we can specify the `tensorboard_writer` parameter to log both the training loss and validation loss. The log result can then be inspected using `tensorboard`.

Below is an example demonstrating how to integrate `tensorboard_writer` into the :class:`SingleSourcingNeuralController`.

.. doctest::

    >>> import torch
    >>> from idinn.demand import UniformDemand
    >>> from idinn.single_controller.single_neural import SingleSourcingNeuralController
    >>> from torch.utils.tensorboard import SummaryWriter

    >>> single_sourcing_model = SingleSourcingModel(
    ...     lead_time=0,
    ...     holding_cost=5,
    ...     shortage_cost=495,
    ...     batch_size=32,
    ...     init_inventory=10,
    ...     demand_generator=UniformDemand(low=0, high=4),
    ...  )
    >>> single_controller = SingleSourcingNeuralController(
    ...     hidden_layers=[2], activation=torch.nn.CELU(alpha=1)
    ... )
    >>> single_controller.fit(
    ...     sourcing_model=single_sourcing_model,
    ...     sourcing_periods=50,
    ...     validation_sourcing_periods=1000,
    ...     epochs=5000,
    ...     seed=1,
    ...     tensorboard_writer=SummaryWriter(comment="_single_1")
    ... )