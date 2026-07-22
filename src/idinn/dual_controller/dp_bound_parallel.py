import logging
from datetime import datetime
from itertools import product
from typing import List as TypingList
from typing import Optional, Tuple, Union, no_type_check

import numpy as np
import torch
from numba import njit, prange
from tqdm import tqdm

from ..demand import UniformDemand
from ..sourcing_model import DualSourcingModel
from .base import BaseDualController

# Get root logger
logger = logging.getLogger()


# ======================================================================
# Parallel Bellman sweep — single generic kernel for any cycle_length n.
# The pipeline shift after a full cycle is entirely determined by the state and the action it does NOT depend on the demand realisation. 
# Only the final inventory position (ipe_final) depends on the full demand path (dem_0, ..., dem_{n-1}).
# So instead of nested per-period demand loops keyed into a numba Dict, we precompute all demand combinations for one cycle up front and do a single flat loop over them inside the njit kernel, looking the next cycle-start state up by a directly-computed flat array index (row-major, via `sizes`/`strides`) instead of a tuple-keyed Dict.
# State layout : (ip, pipeline[0], ..., pipeline[dim_pipeline-1])
# Action layout: (qr0, qe0, qe1, ..., qe_{n-1})
# ======================================================================


@njit(parallel=True)
def _vf_sweep_parallel(
    states_arr: np.ndarray,     # (N_states, dim_pipeline+1) int64
    sizes: np.ndarray,          # (dim_pipeline+1,) int64 - sizes[0]=ip range size
    strides: np.ndarray,        # (dim_pipeline+1,) int64 - row-major strides
    actions: np.ndarray,        # (N_actions, n+1) int64 - col0=qr0, col1..n=qe_t
    demand_combos: np.ndarray,  # (N_combos, n) int64
    combo_probs: np.ndarray,    # (N_combos,) float64
    ce: float,
    h: float,
    b: float,
    min_ip: int,
    max_ip: int,
    vf_arr: np.ndarray,         # (N_states,) float64 - current value estimate
) -> np.ndarray:
    N = states_arr.shape[0]
    dim_pipeline = sizes.shape[0] - 1
    n = actions.shape[1] - 1
    new_vf = np.empty(N, dtype=np.float64)

    for idx in prange(N):
        state = states_arr[idx]
        best_cost = 1e10

        for a in range(actions.shape[0]):
            qr0 = actions[a, 0]

            immediate_cost = 0.0
            for t in range(n):
                immediate_cost += actions[a, 1 + t] * ce

            expected_cost = 0.0
            action_valid = True

            for c in range(demand_combos.shape[0]):
                ip_acc = state[0]
                total_inv_cost = 0.0
                combo_valid = True


                for t in range(n):
                    qe_t = actions[a, 1 + t]
                    if t < dim_pipeline:
                        arrival = state[1 + t]
                    elif t == dim_pipeline:
                        arrival = qr0
                    else:
                        arrival = 0
                    ip_acc = ip_acc + qe_t + arrival - demand_combos[c, t]
    

                    # Mid-cycle states are not stored in vf - validity is
                    # an explicit signed-bound check, same as the dict
                    # version's min_ip/max_ip test.
                    if t < n - 1:
                        if ip_acc < min_ip or ip_acc > max_ip:
                            combo_valid = False
                            break

                    inv = ip_acc - arrival
                    inv_cost = inv * h if inv >= 0 else -inv * b
                    total_inv_cost += inv_cost

                if not combo_valid:
                    action_valid = False
                    break

                # Encode the resulting cycle-start state 
                ip_offset = ip_acc - min_ip
                if ip_offset < 0 or ip_offset >= sizes[0]:
                    action_valid = False
                    break

                total_idx = ip_offset * strides[0]
                ok = True
                for m in range(dim_pipeline):
                    if m < dim_pipeline - n:
                        v = state[1 + n + m]
                    elif m == dim_pipeline - n:
                        v = qr0
                    else:
                        v = 0
                    if v < 0 or v >= sizes[m + 1]:
                        ok = False
                        break
                    total_idx += v * strides[m + 1]

                if not ok or total_idx < 0 or total_idx >= N:
                    action_valid = False
                    break

                vnext = vf_arr[total_idx]
                if vnext > 1e8:
                    action_valid = False
                    break

                expected_cost += combo_probs[c] * (total_inv_cost + vnext)

            if action_valid:
                total = immediate_cost + expected_cost
                if total < best_cost:
                    best_cost = total

        new_vf[idx] = best_cost

    return new_vf


@njit
def _vf_extract_single(
    states_arr: np.ndarray,
    sizes: np.ndarray,
    strides: np.ndarray,
    actions: np.ndarray,
    demand_combos: np.ndarray,
    combo_probs: np.ndarray,
    ce: float,
    h: float,
    b: float,
    min_ip: int,
    max_ip: int,
    vf_arr: np.ndarray,
    idx: int,
) -> Tuple[float, np.ndarray, bool]:
    """Serial single-state re-evaluation for final policy extraction.

    Identical logic to one `idx` iteration of `_vf_sweep_parallel`, but
    also tracks and returns the best action found (the parallel sweep only
    needs the value, not the argmax).
    """
    state = states_arr[idx]
    dim_pipeline = sizes.shape[0] - 1
    n = actions.shape[1] - 1

    best_cost = 1e10
    best_action = actions[0].copy()
    found = False

    for a in range(actions.shape[0]):
        qr0 = actions[a, 0]

        immediate_cost = 0.0
        for t in range(n):
            immediate_cost += actions[a, 1 + t] * ce

        expected_cost = 0.0
        action_valid = True

        for c in range(demand_combos.shape[0]):
            ip_acc = state[0]
            total_inv_cost = 0.0
            combo_valid = True

            for t in range(n):
                qe_t = actions[a, 1 + t]
                if t < dim_pipeline:
                    arrival = state[1 + t]
                elif t == dim_pipeline:
                    arrival = qr0
                else:
                    arrival = 0
                ip_acc = ip_acc + qe_t + arrival - demand_combos[c, t]

                if t < n - 1:
                    if ip_acc < min_ip or ip_acc > max_ip:
                        combo_valid = False
                        break

                inv = ip_acc - arrival
                inv_cost = inv * h if inv >= 0 else -inv * b
                total_inv_cost += inv_cost

            if not combo_valid:
                action_valid = False
                break

            ip_offset = ip_acc - min_ip
            if ip_offset < 0 or ip_offset >= sizes[0]:
                action_valid = False
                break

            total_idx = ip_offset * strides[0]
            ok = True
            for m in range(dim_pipeline):
                if m < dim_pipeline - n:
                    v = state[1 + n + m]
                elif m == dim_pipeline - n:
                    v = qr0
                else:
                    v = 0
                if v < 0 or v >= sizes[m + 1]:
                    ok = False
                    break
                total_idx += v * strides[m + 1]

            if not ok or total_idx < 0 or total_idx >= vf_arr.shape[0]:
                action_valid = False
                break

            vnext = vf_arr[total_idx]
            if vnext > 1e8:
                action_valid = False
                break

            expected_cost += combo_probs[c] * (total_inv_cost + vnext)

        if action_valid:
            total = immediate_cost + expected_cost
            if total < best_cost:
                best_cost = total
                best_action = actions[a]
                found = True

    return best_cost, best_action, found


class DynamicProgrammingController(BaseDualController):
    """
    Cycle-unrolled dynamic programming controller.

    One Bellman update covers a full cycle: period 0 places a regular +
    expedited order, periods 1..N-1 place expedited-only orders. Optimized
    with a flat-array `@njit(parallel=True)` state sweep (see module
    docstring) — no numba Dict / tuple-keyed lookups in the hot loop.

    State layout : (ip, pipeline[0], ..., pipeline[dim_pipeline-1])
    Action layout: (qr0, qe0, qe1, ..., qe_{N-1})
    """

    def __init__(self, cycle_length: int = 2) -> None:
        if cycle_length not in (1, 2, 3):
            raise ValueError("cycle_length must be 1, 2, or 3")
        self.cycle_length = cycle_length
        self.sourcing_model = None
        self.qf = None
        self.vf = None
        logger.info(
            f"Initialized DynamicProgrammingController (parallel) "
            f"with cycle_length={cycle_length}"
        )

    @staticmethod
    def _get_free_slot_indices(lr, cycle_length):
        """
        Return pipeline slot indices that can be non-zero at cycle-start states.
        Rule: state[k] can be non-zero iff k ≡ lr (mod cycle_length).
        """
        target = lr % cycle_length
        return [k for k in range(1, lr) if k % cycle_length == target]

    @staticmethod
    def _get_basestock_ub(
        exp_demand: float, lead_time: int, support: float, h: float, b: float
    ) -> float:
        """
        Get an upper bound on the single-source basestock level based on
        Hoeffding's inequality.
        """
        n = lead_time + 1
        base_stock_ub = n * exp_demand + support * np.sqrt(n * np.log(1 + b / h) / 2)
        return np.ceil(base_stock_ub)

    @no_type_check
    def fit(
        self,
        sourcing_model: DualSourcingModel,
        max_iterations: int = 1000000,
        tolerance: float = 10e-8,
        validation_freq: int = 100,
        log_freq: int = 100,
        bound_slack: int = 1,
    ) -> None:
        """
        Fit the controller to the given sourcing model.

        Parameters
        ----------
        sourcing_model : DualSourcingModel
            The sourcing model to fit the controller to.
        max_iterations : int, default is 1000000
            Specifies the maximum number of iterations to run.
        tolerance : float, default is 10e-8
            Specifies the tolerance to check if the value function has converged.
        validation_freq : int, default is 100
            Specifies how many iterations to run before checking the tolerance.
        log_freq : int, default is 100
            Specifies how many training epochs to run before logging.
        bound_slack : int, default is 1
            Multiplier for widening the IP box in truncation-sensitivity checks.
            Run once with 1 and once with 2; matching values certify that the
            truncation is non-binding.
        """
        self.sourcing_model = sourcing_model

        if not isinstance(sourcing_model.demand_generator, UniformDemand):
            raise ValueError(
                "DynamicProgrammingController only supports uniform demand distribution."
            )
        if sourcing_model.expedited_lead_time != 0:
            raise ValueError(
                "DynamicProgrammingController only supports expedited_lead_time = 0."
            )

        start_time = datetime.now()
        logger.info(f"Starting parallel dynamic programming at {start_time}")
        logger.info(
            f"Sourcing model parameters: batch_size={sourcing_model.batch_size}, "
            f"lead_time={sourcing_model.lead_time}, "
            f"init_inventory={sourcing_model.init_inventory.int().item()}, "
            f"demand_generator={sourcing_model.demand_generator.__class__.__name__}"
        )
        logger.info(
            f"Training parameters: max_iterations={max_iterations}, tolerance={tolerance}, "
            f"cycle_length={self.cycle_length}"
        )

        n = self.cycle_length
        min_demand = int(sourcing_model.demand_generator.get_min_demand())
        max_demand = int(sourcing_model.demand_generator.get_max_demand())
        exp_demand = (max_demand + min_demand) / 2.0
        support = max_demand - min_demand
        h = sourcing_model.get_holding_cost()
        b = sourcing_model.get_shortage_cost()
        ce = sourcing_model.get_expedited_order_cost()
        le = sourcing_model.get_expedited_lead_time()
        lr = sourcing_model.get_regular_lead_time()

        base_e = self._get_basestock_ub(
            exp_demand=exp_demand, lead_time=le, support=support, h=h, b=b
        )
        base_r = self._get_basestock_ub(
            exp_demand=exp_demand, lead_time=lr, support=support, h=h, b=b
        )

        # Original single-period bounds - used only to size the action grid.
        min_ip_single = int(min(base_r, base_e) - max_demand)
        max_ip_single = int(max(base_r, base_e))
        max_order_single = max_ip_single + min_ip_single

        # Cyclic scaling: regular order covers N periods, expedited stays per-period.
        max_order_regular = int(n * max_order_single)
        max_order_expedited = max_order_single

        # Signed IP box (decoupled from base-stock arithmetic). bound_slack
        # widens both sides for truncation-sensitivity re-runs.
        min_ip = -bound_slack * (n + 1) * max_demand
        max_ip = int(max(base_r * n, base_e)) + bound_slack * max_demand

        pipeline_ub = max(int(max_demand), max_order_regular)
        dim_pipeline = lr - le - 1  # le == 0

        logger.info(
            f"Bounds: max_order_regular={max_order_regular}, "
            f"max_order_expedited={max_order_expedited}, "
            f"min_ip={min_ip}, max_ip={max_ip}, pipeline_ub={pipeline_ub}, "
            f"dim_pipeline={dim_pipeline}"
        )

        free_slots = self._get_free_slot_indices(lr, n)
        logger.info(f"Free pipeline slots: {free_slots} (of {list(range(1, lr))})")

        # --- sizes / strides for flat, row-major state indexing -----------
        # dim 0 = ip; dims 1..dim_pipeline = pipeline slots 1..lr-1 in order.
        sizes = np.empty(dim_pipeline + 1, dtype=np.int64)
        sizes[0] = max_ip - min_ip + 1
        for j, k in enumerate(range(1, lr)):
            sizes[j + 1] = pipeline_ub + 1 if k in free_slots else 1

        strides = np.empty(dim_pipeline + 1, dtype=np.int64)
        strides[-1] = 1
        for d in range(dim_pipeline - 1, -1, -1):
            strides[d] = strides[d + 1] * sizes[d + 1]

        # --- Enumerate cycle-start states in the SAME row-major order ------
        # (must match the sizes/strides scheme above exactly, since vf_arr[i]
        # is looked up by array position, not by re-encoding the state).
        slot_ranges = [
            range(pipeline_ub + 1) if k in free_slots else [0]
            for k in range(1, lr)
        ]
        states_list = list(product(range(min_ip, max_ip + 1), *slot_ranges))
        N_states = len(states_list)
        logger.info(f"Reachable state space: {N_states:,}")
        states_arr = np.array(states_list, dtype=np.int64)

        # --- Actions: (qr0, qe0, qe1, ..., qe_{n-1}) ------------------------
        actions_list = list(product(
            range(max_order_regular + 1),
            *([range(max_order_expedited + 1)] * n),
        ))
        actions_arr = np.array(actions_list, dtype=np.int64)
        logger.info(
            f"State space size: {N_states}, Action space size: {len(actions_arr)}"
        )

        # --- Demand combinations across the n periods of one cycle ---------
        # (precomputed on the Python side — no numba Dict / typed containers
        # needed anywhere in the hot loop anymore.)
        demand_prob_map = sourcing_model.demand_generator.enumerate_support()
        demand_vals_1p = np.array(sorted(demand_prob_map.keys()), dtype=np.int64)
        demand_probs_1p = np.array(
            [demand_prob_map[k] for k in demand_vals_1p], dtype=np.float64
        )

        combo_idx = list(product(range(len(demand_vals_1p)), repeat=n))
        demand_combos = np.array(
            [[demand_vals_1p[i] for i in combo] for combo in combo_idx],
            dtype=np.int64,
        )
        combo_probs = np.array(
            [
                float(np.prod([demand_probs_1p[i] for i in combo]))
                for combo in combo_idx
            ],
            dtype=np.float64,
        )

        # --- Value iteration (parallel sweep) -------------------------------
        vf_arr = np.ones(N_states, dtype=np.float64)
        all_values = np.zeros(max_iterations, dtype=np.float64)
        val = 0.0

        logger.info("Starting value iteration (parallel sweep)...")
        for iteration in tqdm(range(max_iterations)):
            vf_arr = _vf_sweep_parallel(
                states_arr, sizes, strides,
                actions_arr, demand_combos, combo_probs,
                ce, h, b, min_ip, max_ip,
                vf_arr,
            )

            finite_vals = vf_arr[vf_arr < 1e9]
            this_average = np.mean(finite_vals) if finite_vals.size > 0 else 0.0
            val = this_average / (iteration + 1)
            all_values[iteration] = val

            if iteration > 1 and iteration % log_freq == 0:
                logger.info(
                    f"Epoch {iteration}/{max_iterations} - Value: {val:.4f}"
                )

            if iteration > 1 and iteration % validation_freq == 0:
                delta = all_values[iteration - 1] - all_values[iteration]
                if abs(delta) <= tolerance:
                    logger.info(f"Converged at iteration {iteration}")
                    break

        # --- Policy extraction (serial) --------------------------------------
        logger.info("Extracting policy...")
        qf = {}
        n_sat_r = 0
        n_sat_e = 0
        for i, s in enumerate(states_list):
            cost, best_action, found = _vf_extract_single(
                states_arr, sizes, strides, actions_arr,
                demand_combos, combo_probs,
                ce, h, b, min_ip, max_ip,
                vf_arr, i,
            )
            if found:
                action_tuple = tuple(int(v) for v in best_action)
                qf[s] = action_tuple
                if action_tuple[0] == max_order_regular:
                    n_sat_r += 1
                if any(q == max_order_expedited for q in action_tuple[1:]):
                    n_sat_e += 1

        self.qf = qf
        self.vf = val

        # --- Post-fit diagnostics: warn if the action grid is saturated -----
        if n_sat_r > 0 or n_sat_e > 0:
            logger.warning(
                f"Action grid saturated: {n_sat_r} states at max_order_regular, "
                f"{n_sat_e} states at max_order_expedited. Widen the action grid "
                f"and re-run."
            )

        end_time = datetime.now()
        duration = end_time - start_time
        logger.info(f"Dynamic programming completed at {end_time}")
        logger.info(f"Total training duration: {duration}")
        logger.info(
            f"Final best cost: "
            f"{self.get_average_cost(self.sourcing_model, sourcing_periods=1000, seed=42):.4f}"
        )

    def predict(
        self,
        current_inventory: Union[int, torch.Tensor],
        past_regular_orders: Optional[Union[TypingList[int], torch.Tensor]] = None,
        past_expedited_orders: Optional[Union[TypingList[int], torch.Tensor]] = None,
        output_tensor: bool = False,
    ) -> Union[Tuple, Tuple[int, ...]]:
        """
        Parameters
        ----------
        current_inventory : int, or torch.Tensor
            Current inventory.
        past_regular_orders : list, or torch.Tensor, optional
            Past regular orders. If the length of `past_regular_orders` is lower than
            `regular_lead_time`, it will be padded with zeros.
        past_expedited_orders : list, or torch.Tensor, optional
            Ignored. Expedited lead time is assumed to be 0.
        output_tensor : bool, default is False
            If True, returns a tuple of torch.Tensors of length cycle_length + 1
            in the order (qr0, qe0, qe1, ..., qe_{N-1}).
        """
        if self.sourcing_model is None:
            raise AttributeError("The controller is not trained.")

        regular_lead_time = self.sourcing_model.get_regular_lead_time()

        current_inventory = self._check_current_inventory(current_inventory)
        past_regular_orders = self._check_past_orders(
            past_regular_orders, regular_lead_time
        )

        first = (
            current_inventory.squeeze()
            + past_regular_orders.squeeze()[-regular_lead_time]
        )
        second = past_regular_orders.squeeze()[-regular_lead_time + 1:]
        key = tuple([int(first)] + second.int().tolist())

        if output_tensor:
            return tuple(torch.tensor([[v]]) for v in self.qf[key])
        return self.qf[key]

    def get_last_cost(self, sourcing_model: DualSourcingModel) -> torch.Tensor:
        """Calculate the cost for the latest period."""
        last_regular_q = sourcing_model.get_last_regular_order()
        last_expedited_q = sourcing_model.get_last_expedited_order()
        regular_order_cost = sourcing_model.get_regular_order_cost()
        expedited_order_cost = sourcing_model.get_expedited_order_cost()
        holding_cost = sourcing_model.get_holding_cost()
        shortage_cost = sourcing_model.get_shortage_cost()
        current_inventory = sourcing_model.get_current_inventory()
        last_cost = (
            regular_order_cost * last_regular_q
            + expedited_order_cost * last_expedited_q
            + holding_cost * torch.relu(current_inventory)
            + shortage_cost * torch.relu(-current_inventory)
        )
        return last_cost

    @no_type_check
    def get_total_cost(
        self,
        sourcing_model: DualSourcingModel,
        sourcing_periods: int,
        seed: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Calculate the total cost over `sourcing_periods` cycles.

        Each cycle covers `cycle_length` real periods: one regular + expedited
        order in period 0, then expedited-only orders in periods 1..N-1.
        """
        sourcing_model.reset()
        if seed is not None:
            torch.manual_seed(seed)

        total_cost = torch.tensor(0.0)
        for _ in tqdm(range(sourcing_periods)):
            current_inventory = sourcing_model.get_current_inventory()
            past_regular_orders = sourcing_model.get_past_regular_orders()
            past_expedited_orders = sourcing_model.get_past_expedited_orders()

            # actions = (qr0, qe0, qe1, ..., qe_{N-1})
            actions = self.predict(
                current_inventory,
                past_regular_orders,
                past_expedited_orders,
                output_tensor=True,
            )

            # Period 0: regular + first expedited order
            sourcing_model.order(actions[0], actions[1])
            total_cost += self.get_last_cost(sourcing_model).mean()

            # Periods 1..N-1: expedited only
            for t in range(1, self.cycle_length):
                sourcing_model.order(torch.zeros_like(actions[0]), actions[t + 1])
                total_cost += self.get_last_cost(sourcing_model).mean()

        return total_cost

    @no_type_check
    def get_average_cost(
        self,
        sourcing_model: DualSourcingModel,
        sourcing_periods: int,
        seed: Optional[int] = None,
    ) -> torch.Tensor:
        """Calculate the average per-period cost."""
        return (
            self.get_total_cost(sourcing_model, sourcing_periods, seed)
            / (sourcing_periods * self.cycle_length)
        )

    def reset(self) -> None:
        self.qf = None
        self.vf = None
        self.sourcing_model = None