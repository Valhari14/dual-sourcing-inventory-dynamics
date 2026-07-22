Introduction
============

Dual-sourcing problems involve managing inventory with two suppliers, each having different lead times (how long it takes for orders to arrive) and order costs (the cost of placing an order). The main challenge is to decide which supplier to use for each order to minimize costs while dealing with unpredictable demand.

Key Concepts
------------

- **Holding Cost:** The cost incurred for keeping excess inventory. The more inventory you have, the higher the total holding cost.
- **Shortage Cost:** The penalty for not having enough inventory to meet demand. The higher the unit shortage cost, the more critical it is to avoid stockouts.
- **Demand:** Demand is a stochastic variable, requiring careful planning.
- **Regular and Expedited Suppliers:** Two types of suppliers with different lead times and order costs. The regular supplier typically has a longer lead time but lower cost, while the expedited supplier has a shorter lead time but higher cost.

Notation
--------

We use the following notation to describe the problem:

- :math:`I_t`: Net inventory before replenishment in period :math:`t`.
- :math:`D_t`: Demand in period :math:`t`.
- :math:`b`: Shortage cost per unit of inventory.
- :math:`h`: Holding cost per unit of inventory.
- :math:`q^{\rm r}_t, q^{\rm e}_t`: Quantities ordered from the regular and expedited suppliers in period :math:`t`, respectively.
- :math:`c_{\rm r}, c_{\rm e}`: Ordering costs from the regular and expedited suppliers, respectively.
- :math:`l_{\rm r}, l_{\rm e}`: Lead times of the regular and expedited suppliers, respectively.

Dual-Sourcing Dynamics
----------------------

The sequence of events in a single period :math:`t` is as follows:

1. Order quantities :math:`q^{\rm r}_{t-l_{\rm r}}` and :math:`q^{\rm e}_{t-l_{\rm e}}`, ordered in periods :math:`t-l_{\rm r}` and :math:`t-l_{\rm e}`, arrive.
2. Order quantities :math:`q^{\rm r}_t` and :math:`q^{\rm e}_t` are placed.
3. Demand :math:`D_t` is realized.
4. Inventory cost for the period is registered as :math:`c_{\rm r} q^{\rm r}_t + c_{\rm e} q^{\rm e}_t + h[I_{t} + q^{\rm r}_{t-l_{\rm r}} + q^{\rm e}_{t-l_{\rm e}} - D_t]^+ + b[I_{t} + q^{\rm r}_{t-l_{\rm r}} + q^{\rm e}_{t-l_{\rm e}} - D_t]^+`, where :math:`[x]^+ = \max\{0, x\}`.
5. New state is updated as :math:`(I_{t} + q^{\rm r}_{t-l_{\rm r}} + q^{\rm e}_{t-l_{\rm e}} - D_t, q^{\rm r}_{t-l_{\rm r}+1}, \dots, q^{\rm r}_{t}, q^{\rm e}_{t-l_{\rm e}+1}, \dots, q^{\rm e}_{t})`.

The net inventory evolves according to

.. math::

   I_{t+1} = I_{t} + q^{\rm r}_{t-l_{\rm r}} + q^{\rm e}_{t-l_{\rm e}} - D_t \,,

and the cost at period :math:`t`, :math:`c_t`, is

.. math::

   c_t = c_{\rm r} q^{\rm r}_t + c_{\rm e} q^{\rm e}_t + h \max\{0, I_{t+1}\} + b \max\{0, -I_{t+1}\}\,.

The higher the holding cost, the more costly it is to keep the inventory positive and high. The higher the shortage cost, the more costly it is to run out of stock when the inventory level is negative. The higher the regular and expedited order costs, the more costly it is to place the respective orders.

Available Controllers
---------------------

- **CappedDualIndexController:** The Capped Dual Index controller uses a capped dual index policy to determine order quantities, balancing the trade-off between holding and shortage costs.
- **DynamicProgrammingController:** The Dynamic Programming controller applies dynamic programming techniques to solve the dual-sourcing problem optimally by considering all possible future states and decisions.
- **DualSourcingNeuralController:** The neural controller ia based on a neural network that outputs orders given the current system state. The neural network is trained by minimizing the total cost over time.


This documentation provides a introduction for understanding dual-sourcing problems and their solutions. For more details on using these controllers, refer to the following sections.

References
----------

- Scarf, H., & Karlin, S. (1958). Inventory models of the Arrow-Harris-Marschak type with time lag. In K. J. Arrow, S. Karlin, & H. E. Scarf (Eds.), *Studies in the Mathematical Theory of Inventory and Production* (Stanford University Press, Stanford, CA).
- Arrow, K. J., Harris, T., & Marschak, J. (1951). Optimal inventory policy. *Econometrica*, 19(3), 250–272.