Single-Sourcing Neural Network Controller
=========================================

Instead of relying on a traditional base-stock controller, actions in a single-sourcing inventory system can be guided by a neural network. This modern approach leverages neural networks and automatic differentiation to create a system capable of making dynamics-informed decisions for inventory management. 

Key Concepts
------------

- **Neural Network**: Instead of relying on fixed rules like the base-stock policy, actions are parameterized through a neural network that adapts based on training data.

- **Discrete-Time Dynamics**: The system evolves over discrete time intervals, and the controller must optimize actions at each step to reduce long-term costs.

- **Training and Optimization**: The neural network aims to minimize the expected per-period cost, balancing factors such as lead time, holding costs, shortage costs, past orders and inventory level. It is trained using simulated data provided by the sourcing mode to learn optimal policies for varying demand patterns and lead times.

For further details, see Böttcher, Asikis, and Fragkos (2023).

Example Usage
-------------

We now present an example to demonstrate how the :class:`SingleSourcingNeuralController` can be called, trained, and evaluated in `idinn`.

.. doctest::
    
    >>> from idinn.sourcing_model import SingleSourcingModel
    >>> from idinn.single_controller import SingleSourcingNeuralController
    >>> from idinn.demand import UniformDemand
    >>> from torch.utils.tensorboard import SummaryWriter

    >>> single_sourcing_model = SingleSourcingModel(
    ...     lead_time=0,
    ...     holding_cost=5,
    ...     shortage_cost=495,
    ...     batch_size=32,
    ...     init_inventory=10,
    ...     demand_generator=UniformDemand(low=0, high=4),
    ... )
    >>> controller_neural = SingleSourcingNeuralController()
    >>> controller_neural.fit(
    ...     sourcing_model=single_sourcing_model,
    ...     sourcing_periods=50,
    ...     validation_sourcing_periods=1000,
    ...     epochs=2000,
    ...     tensorboard_writer=SummaryWriter(comment="_single_1"),
    ...     seed=1,
    ... )
    >>> # Avg. cost near 10
    >>> avg_cost = controller_neural.get_average_cost(single_sourcing_model, sourcing_periods=1000)
    >>> print(f"Average cost: {avg_cost:.2f}")
    Average cost: 10.00

Adjusting parameters such as `batch_size`, `init_inventory`, and `epochs` can improve the learning of sourcing policies.

For a given controller, orders can be predicted as follows if the lead time is 0.

.. doctest::

    >>> controller_neural.predict(current_inventory=10)

If the lead-time value is greater than 0, one has to specify the corresponding `past_orders`, otherwise the past orders are assumed to be 0.

.. doctest::

    >>> controller_neural.predict(current_inventory=10, past_orders=[9, 11, 12])

References
----------
- Böttcher, L., Asikis, T., & Fragkos, I. (2023). Control of dual-sourcing inventory systems using recurrent neural networks. *INFORMS Journal on Computing*, 35(6), 1308–1328.