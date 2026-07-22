Capped Dual Index Controller
============================

The capped dual index (CDI) policy helps manage inventory by using both regular and expedited orders to meet demand while keeping costs low. Regular orders have a longer lead time and are capped at a maximum limit, while expedited orders have a shorter lead time and address immediate needs. The policy uses two target inventory levels, one for each order type, and places orders based on the difference between these targets and the current inventory. This approach balances cost and responsiveness by effectively using both types of orders. The optimal parameters are identified using grid search.

Key Concepts
------------

- **Regular Orders**: Orders that have a longer lead time and are capped at a maximum limit.
- **Expedited Orders**: Orders that have a shorter lead time and are used to address immediate needs.
- **Target Inventory Levels**: The desired inventory levels for both regular and expedited orders.
- **Grid Search**: A method used to find the optimal CDI parameters by searching over specified ranges.


Mathematical Structure
----------------------

The capped dual index policy (Sun & Van Mieghem, 2019) determines the regular and expedited orders in period :math:`t` as follows.

For regular orders, we use

.. math::

   q_t^{\rm r} = \min \left\{ \left[ S_t^{\rm r *} - I_t^{t+l-1} \right]^+, \bar{q}_t^{\rm r *} \right\}\,.

For expedited orders, we use

.. math::

   q_t^{\rm e} = \left[ S_t^{\rm e *} - I_t^t \right]^+\,.

Here, :math:`l_{\rm e} = 0` is assumed. The term :math:`I_t^{t+k}` represents the net inventory at the start of period :math:`t` plus all in-transit orders arriving by period :math:`t+k`. That is,

.. math::

   I_t^{t+k} = I_{t-1} + \sum_{i=t}^{\min(t+k, t-1)} q_i^{\rm e} + \sum_{i=t-l_{\rm r}}^{t-l_{\rm r}+k} q_i^{\rm r}\,,

where :math:`k` ranges from 0 to :math:`l_{\rm r} - 1`. According to Sun & Van Mieghem (2019), if :math:`a > b`, then :math:`\sum_{i=a}^b = 0`. The parameters :math:`(S_t^{\rm r *}, S_t^{\rm e *}, \bar{q}_t^{\rm r *})` are determined through a search procedure. If the demand distribution is constant over time, the CDI parameters simplify to :math:`S_t^{\rm r *} = S^{\rm r *}`, :math:`S_t^{\rm e *} = S^{\rm e *}`, and :math:`\bar{q}_t^{\rm r *} = \bar{q}^{\rm r *}`.

Example Usage
-------------

We now present one example to demonstrate how the :class:`CappedDualIndexController` can be called, trained, and evaluated in `idinn`.

.. doctest::
    
   >>> from idinn.sourcing_model import DualSourcingModel
   >>> from idinn.dual_controller import CappedDualIndexController
   >>> from idinn.demand import UniformDemand

   >>> dual_sourcing_model = DualSourcingModel(
   ...    regular_lead_time=2,
   ...    expedited_lead_time=0,
   ...    regular_order_cost=0,
   ...    expedited_order_cost=20,
   ...    holding_cost=5,
   ...    shortage_cost=495,
   ...    init_inventory=0,
   ...    demand_generator=UniformDemand(low=0, high=4)
   ... )
   >>> controller_cdi = CappedDualIndexController()
   >>> controller_cdi.fit(
   ...    dual_sourcing_model,
   ...    sourcing_periods=100
   ... )
   >>> controller_cdi.get_average_cost(dual_sourcing_model, sourcing_periods=1000)  # doctest: +ELLIPSIS
   25.56

Adjusting the `sourcing_periods` parameter in `controller_cdi` can improve the controller's performance. Additionally, the `fit` function provides parameters such as `s_e_range`, `s_r_range`, and `q_r_range` to define the ranges of CDI parameters for the grid search. By default, all these ranges are set to `np.arange(2, 11)`.

References
----------

- Sun, J., & Van Mieghem, J. A. (2019). Robust dual sourcing inventory management: Optimality of capped dual index policies and smoothing. *Manufacturing & Service Operations Management*, 21(4), 912–931.