Base-Stock Controller
=====================

For single-sourcing problems over an infinite time horizon, the base-stock controller maintains a fixed target inventory level. When inventory falls below this level due to demand fluctuations, a replenishment order is immediately placed to restore the stock to the target level.

Key Concepts
------------

- **Target Inventory Level:** The target inventory level that triggers replenishment orders whenever stock dips below it.

Mathematical Structure
----------------------

To mathematically describe the optimal order policy of single-sourcing problems (Arrow, Harris, & Marschak, 1951; Scarf & Karlin, 1958), we use :math:`l` and :math:`z^*` to respectively denote the replenishment lead time and the target inventory-position level (i.e., the target net inventory level plus all items ordered but not received yet). The inventory position of single-sourcing dynamics at time :math:`t`, :math:`\tilde{I}_t`, is given by

.. math::

   \tilde{I}_t =
   \begin{cases}
      I_t & \text{if} \,\, l=0 \\
      I_t + \sum_{i=1}^l q_{t-i} & \text{if} \,\, l>0 \,,
   \end{cases}

where :math:`I_t` and :math:`q_t` denote the net inventory at time :math:`t` and the replenishment order placed at time :math:`t`, respectively. 

The optimal order quantity is :math:`q_t=\max\{0, z^*-\tilde{I}_t\}`, where the target level :math:`z^*` is the parameter to be determined.

Example Usage
-------------

We now present two examples to demonstrate how the :class:`BaseStockController` can be called and evaluated in `idinn`.

.. doctest::

   >>> import torch
   >>> from idinn.sourcing_model import SingleSourcingModel
   >>> from idinn.single_controller import BaseStockController
   >>> from idinn.demand import UniformDemand

   >>> # First example
   >>> single_sourcing_model = SingleSourcingModel(
   ...     lead_time=0,
   ...     holding_cost=5,
   ...     shortage_cost=495,
   ...     init_inventory=10,
   ...     demand_generator=UniformDemand(low=0, high=4),
   ... )
   >>> controller_base = BaseStockController()
   >>> # z_star should be 4
   >>> controller_base.fit(single_sourcing_model)
   >>> print(f"z_star: {controller_base.z_star}")
   z_star: 4
   >>> # Avg. cost near 10
   >>> avg_cost = controller_base.get_average_cost(single_sourcing_model, sourcing_periods=1000, seed=42)
   >>> print(f"Average Cost: {avg_cost:.2f}")
   Average Cost: 10.39

   >>> # Second example
   >>> single_sourcing_model = SingleSourcingModel(
   ...     lead_time=2,
   ...     holding_cost=5,
   ...     shortage_cost=495,
   ...     init_inventory=10,
   ...     demand_generator=UniformDemand(low=0, high=4),
   ... )
   >>> controller_base = BaseStockController()
   >>> controller_base.fit(single_sourcing_model)
   >>> # z_star should be 11
   >>> print(f"z_star: {controller_base.z_star}")
   z_star: 11
   >>> # Avg. cost near 29
   >>> avg_cost = controller_base.get_average_cost(single_sourcing_model, sourcing_periods=1000, seed=42)
   >>> print(f"Average Cost: {avg_cost:.2f}")
   Average Cost: 29.53

References
----------
- Scarf, H., & Karlin, S. (1958). Inventory models of the Arrow-Harris-Marschak type with time lag. In K. J. Arrow, S. Karlin, & H. E. Scarf (Eds.), *Studies in the Mathematical Theory of Inventory and Production* (Stanford University Press, Stanford, CA).
- Arrow, K. J., Harris, T., & Marschak, J. (1951). Optimal inventory policy. *Econometrica*, 19(3), 250–272.
