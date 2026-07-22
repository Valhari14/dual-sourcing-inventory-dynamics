######################################################
idinn: Inventory-Dynamics Control with Neural Networks
######################################################

..  youtube:: hUBfTWV6tWQ
   :width: 100%

`idinn` implements inventory dynamics–informed neural network and other related controllers for solving single-sourcing and dual-sourcing problems. Controllers and inventory dynamics are implemented into customizable objects using PyTorch as backend to enable users to find the optimal controllers for the user-specified inventory systems.

Demo
====

For a quick demo, you can run our `Streamlit app`_. The app allows you to interactively train and evaluate neural controllers for user-specified dual-sourcing systems. 

Alternatively, refer to our Jupyter notebook in `Colab`_.

.. _Streamlit app: https://idinn-demo.streamlit.app/
.. _Colab: https://colab.research.google.com/drive/1BAMiveGXmErIp10MK3V_SUJlDAXHAyaI


Example Usage
=============

.. doctest:: python

   >>> from idinn.sourcing_model import SingleSourcingModel
   >>> from idinn.single_controller import BaseStockController
   >>> from idinn.demand import UniformDemand

   >>> # Initialize the sourcing model
   >>> single_sourcing_model = SingleSourcingModel(
   ...     lead_time=0,
   ...     holding_cost=5,
   ...     shortage_cost=495,
   ...     batch_size=32,
   ...     init_inventory=10,
   ...     demand_generator=UniformDemand(low=1, high=4),
   ... )

   >>> # Initialize a controller
   >>> controller = BaseStockController()

   >>> # Train the controller
   >>> controller.fit(
   ...     sourcing_model=single_sourcing_model,
   ...     seed=42,
   ... )

   >>> # Simulate and plot the results
   >>> controller.plot(sourcing_model=single_sourcing_model, sourcing_periods=100) # doctest: +SKIP

   >>> # Calculate the optimal order quantity for applications
   >>> controller.predict(current_inventory=10, past_orders=[1, 5])
   0


.. toctree::
   :hidden:
   :maxdepth: 1
   :caption: Get Started

   get_started/installation
   get_started/get_started
   get_started/deployment

.. toctree::
   :hidden:
   :maxdepth: 1
   :caption: Single-Sourcing Problems

   single_controller/intro_single_sourcing
   single_controller/base_stock
   single_controller/single_neural

.. toctree::
   :hidden:
   :maxdepth: 1
   :caption: Dual-Sourcing Problems

   dual_controller/intro_dual_sourcing
   dual_controller/capped_dual_index
   dual_controller/dynamic_programming
   dual_controller/dual_neural

.. toctree::
   :hidden:
   :maxdepth: 1
   :caption: Utilities

   utilities/custom_demand
   utilities/save_load
   utilities/plot
   utilities/tensorboard

.. toctree::
   :hidden:
   :maxdepth: 1
   :caption: References

   utilities/api