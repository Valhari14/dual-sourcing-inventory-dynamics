Introduction
============

Single-sourcing problems are common in inventory management and involve finding the optimal order quantities to minimize costs associated with holding excess inventory and experiencing shortages facing stochastic demands. This guide explains the basic concepts and introduces controllers available in ``idinn`` to solve these problems.

Key Concepts
------------

- **Holding Cost:** The cost incurred for keeping excess inventory. The more inventory you have, the higher the total holding cost.
- **Shortage Cost:** The penalty for not having enough inventory to meet demand. The higher the unit shortage cost, the more critical it is to avoid stockouts.
- **Demand:** Demand is a stochastic variable, requiring careful planning.

Notation
--------

We use the following notation to describe the problem:

- :math:`I_t`: Inventory level before replenishment in period :math:`t`.
- :math:`D_t`: Demand in period :math:`t`.
- :math:`q_t`: Quantity ordered from the supplier in period :math:`t`.
- :math:`l`: Supplier's lead time.
- :math:`b`: Shortage cost per unit of inventory.
- :math:`h`: Holding cost per unit of inventory.

Single-Sourcing Dynamics
------------------------

The sequence of events in each period is as follows:

1. Order placed in period :math:`t-l` arrives.
2. New order :math:`q_t` is placed.
3. Demand :math:`D_t` is realized.
4. Inventory cost is registered.
5. Inventory state is updated.

The net inventory evolves according to

.. math::

   I_{t+1} = I_{t} + q_{t-l} - D_t \,.

The cost incurred at period :math:`t` is

.. math::

   c_t = h \max\{0, I_{t+1}\} + b \max\{0, -I_{t+1}\} \,.

Here, :math:`\max\{0, I_{t+1}\}` quantifies the potential excess inventory, and :math:`\max\{0, -I_{t+1}\}` quantifies the potential inventory shortage. The goal is to minimize these costs summed over time.

Available Controllers
---------------------

- **BaseStockController:** The base-stock policy is a widely used approach in inventory management that aims to maintain a consistent inventory level by ordering enough stock to replenish the expected demand, which is calculated based on simulated historical demands in ``idinn``. 
- **SingleSourcingNeuralController:** The neural controller ia based on a neural network that outputs an order given the current system state. The neural network is trained by minimizing the total cost over time.

This introduction provides a foundation for understanding single-sourcing problems and their solutions. For more details on using the described controllers, refer to the following sections.

