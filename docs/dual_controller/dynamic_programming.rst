Dynamic Programming Controller
==============================

Dual-sourcing problems can be formulated and solved via dynamic programming, using the Bellman equation. This approach involves breaking down the problem into smaller subproblems and solving them recursively. However, due to the curse of dimensionality and the computational constraints, where the state and action spaces grow exponentially with the number of variables, the ability of this approach to solve large-scale problems is limited.

Key Concepts
------------

- **Dynamic Programming**: A method for solving complex problems by breaking them down into simpler subproblems. It is applicable where the problem can be divided into overlapping subproblems that can be solved independently.
- **Bellman Equation**: A recursive equation that provides a way to calculate the optimal policy by solving subproblems and combining their solutions. It is central to dynamic programming and reinforcement learning.
- **State Space**: The set of all possible states in which the system can exist. In the context of dual-sourcing, it includes inventory levels and outstanding orders.
- **Action Space**: The set of all possible actions that can be taken in each state. For dual-sourcing, this includes the quantities to order from regular and expedited sources.
- **Cost Function**: A function that assigns a cost to each state-action pair. It typically includes holding costs, shortage costs, and ordering costs.
- **Transition Probabilities**: The probabilities of moving from one state to another given a particular action. These are determined by the demand distribution and lead times.
- **Value Iteration**: An algorithm for solving the Bellman equation by iteratively updating the value of each state until convergence.

Mathematical Structure
----------------------

The current implementation assumes :math:`l_{\rm e}=c_{\rm r}=0`. Note that :math:`c_{\rm r}=0` can be assumed without loss of generality, while :math:`l_{\rm e}=0` implies `some` loss of generality, but allows our implementation to be more versatile because otherwise we would need an overhead of calculating demand convolutions. As such, we hereafter set :math:`l_{\rm r}=l` for notational convenience.

In order to introduce the Bellman equation, we further define the following quantities:

- **Expedited inventory position:** :math:`I_t^{\rm e}=I_t+q_{t-l}`.
- **States:** :math:`\mathbf{s}_t=(I_t^{\rm e}, q^r_{t-l+1}, \dots, q^r_{t-1})`. The state space is denoted as the set of feasible states, :math:`\mathcal{S}=\{\mathbf{s}\}`.
- **Actions:** :math:`\mathbf{Q}=(q^{\rm r},q^{\rm e})`. We define the action space as :math:`\mathcal{D}_\mathbf{Q}:=\{\mathbf{Q}\}`.
- **Cost:** The cost function is :math:`f(x)=b[-x]^++h[x]^+`, where :math:`[x]^+=\max\{x,0\}`.

.. note::
   In theory, both the state and the action spaces are infinitely countable. For example, the policy of not placing any orders will cause the state to approach :math:`(-\infty, 0, \dots, 0)`. Likewise, since there is no limit on the order sizes, the action space is a subset of :math:`\mathbb{N}^2`. However, we restrict our attention to static, near-optimal policies, which in the steady state generate an ergodic Markov chain in the state space. This means we can restrict our attention to a finite part of the state and action spaces.

Once we have selected the actions :math:`(q^{\rm r}_t,q^{\rm e}_t)`, random demand :math:`D_t` is realized in period :math:`t`. Then the state :math:`\mathbf{s}_t=(I_t^{\rm e}, q^{\rm r}_{t-l+1}, \dots, q^{\rm r}_{t-1})` undergoes the following transitions:

.. math::

   I^{\rm e}_{t+1} &\leftarrow I^{\rm e}_{t}+q^{\rm e}_t+q^{\rm r}_{t-l+1}-D_t\\
   q^{\rm r}_{t-l+1} &\leftarrow q^{\rm r}_{t-l+2}\\
   &\dots\\
   q^{\rm r}_{t-2} &\leftarrow q^{\rm r}_{t-1}\\
   q^{\rm r}_{t-1}&\leftarrow q^{\rm r}_t

The corresponding Bellman equation is

.. math::
   J_{t+1}(\mathbf{s})=\min_{\mathbf{a}_t\in \mathcal{A}_t}\left\{c_t(\mathbf{s}_t,\mathbf{a}_t)+\gamma \sum_{\mathbf{s}'\in\mathcal{S}_{t}} \Pr(\mathbf{s}_{t+1}=\mathbf{s'}|\mathbf{s}_t,\mathbf{a}_t)J_{t}(\mathbf{s}')\right\},\,  \mathbf{s}\in\mathcal{S}.


Using renewal theory, it can be shown that for stationary demand distributions 

.. math::
   J^*=\lim\limits_{t\rightarrow \infty}\frac{J_t(\mathbf{s})}{t}\,.


The implemented dynamic-programming controller solves the Bellman equation using value iteration. The iterations are as follows:

1. For each state :math:`\mathbf{s} \in \mathcal{S}`, select an arbitrary initial cost :math:`J_0(\mathbf{s})`.
2. For a given state :math:`\mathbf{s}` and action :math:`\mathbf{Q}`, find the transition probabilities to state :math:`\mathbf{s}'` according to the demand distribution :math:`\phi`. Let us denote those probabilities by :math:`P(\mathbf{s}' | \mathbf{s}, \mathbf{Q})`. Calculate the cost :math:`f(\mathbf{s}')` associated with each transition :math:`\mathbf{s}\xrightarrow{\mathbf{Q}} \mathbf{s}'`. Iterate those calculations for all combinations :math:`(\mathbf{s}, \mathbf{Q}) \in \mathcal{S} \times \mathcal{D}_{\mathbf{Q}}`. 
3. Apply the update

.. math::
   J_{k+1}(\mathbf{s}) = \min\limits_{\mathbf{Q} \in \mathcal{D}_{\mathbf{Q}}} \left\{ c_{\rm e}q^{\rm e} + \sum\limits_{\mathbf{s}' \in \mathcal{S}} P(\mathbf{s}' | \mathbf{s}, \mathbf{Q})(f(\mathbf{s}')+J_{k}(\mathbf{s}')) \right\}\,,\quad \text{for all} \quad \mathbf{s} \in \mathcal{S}\,.

4. Calculate the expected cost approximation :math:`\lambda_{k+1}(\mathbf{s}) = J_{k+1}(\mathbf{s}) / (k+1)`, for all :math:`\mathbf{s} \in \mathcal{S}`.
5. Iterate the above update until :math:`\max\limits_{\mathbf{s}\in\mathcal{S}}\left\{\lvert\lambda_{k+1}(\mathbf{s})-\lambda_{k}(\mathbf{s})\rvert\right\} < \epsilon`.

Example Usage
-------------

We can solve dual-sourcing problems with `idinn` using :class:`DynamicProgrammingController`, which provides a similar interface to other controllers. Note that expedited orders are assumed to have a lead time of 0. The user can increase the `max_iterations` parameter and descrease the `tolerance` to achieve better results, though this will require additional time.

.. doctest::

   >>> from idinn.sourcing_model import DualSourcingModel
   >>> from idinn.dual_controller import DynamicProgrammingController
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
   >>> controller_dp = DynamicProgrammingController()
   >>> controller_dp.fit(
   ...    dual_sourcing_model,
   ...    max_iterations=10000,
   ...    tolerance=1e-6
   ... )
   >>> print(f'Average cost: {controller_dp.vf:.2f}')
   Average cost: 123.45
   >>> print(f'Policy Dictionary: {controller_dp.qf}') # doctest: +ELLIPSIS
   Policy Dictionary: {...}
   >>> # Average cost for a specific trajectory
   >>> controller_dp.get_average_cost(dual_sourcing_model, sourcing_periods=1000)
   123.45
