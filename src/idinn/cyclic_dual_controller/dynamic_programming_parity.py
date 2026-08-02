import logging
from datetime import datetime
from itertools import product
from typing import Dict as TypingDict
from typing import List as TypingList
from typing import Optional, Tuple, Union, no_type_check

import numpy as np
import torch
from numba import njit, prange, types
from numba.typed import Dict, List
from tqdm import tqdm

from ..demand import UniformDemand
from ..sourcing_model import DualSourcingModel
from .base import BaseDualController

logger = logging.getLogger()


@njit(parallel=True)
def _vf_sweep_parallel(
    n: int,
    states_arr: np.ndarray,      # (N_states, state_dim)
    period_flags: np.ndarray,    # (N_states,) 1=period0, 0=other
    actions_0: np.ndarray,       # (N_actions_0, 2)
    actions_t: np.ndarray,       # (N_actions_t, 2)
    demand_vals: np.ndarray,     # (N_demand,)
    demand_probs: np.ndarray,    # (N_demand,)
    ce: float,
    h: float,
    b: float,
    vf_arr: np.ndarray,          # (N_states,) current values
    # index lookup parameters
    n_periods: int,
    ip_min: int,
    ip_range: int,
    max_demand: int,
    dim_pipeline: int,
) -> np.ndarray:
    """
    Parallel Bellman sweep. Uses a flat numpy array for VF lookup
    instead of a numba Dict — avoids all tuple-type issues.

    State encoding: index = t * (ip_range * pipeline_stride)
                           + (ip - ip_min) * pipeline_stride
                           + pipeline_flat_index
    """
    N = states_arr.shape[0]
    new_vf = np.empty(N, dtype=np.float64)

    # Precompute strides for state -> index mapping
    # state layout: (t, ip, p0, p1, ...)
    # pipeline entries each range over [0, max_demand]
    pipeline_size = max_demand + 1  # values 0..max_demand

    # stride for each dimension (row-major)
    # dim order: t, ip, p[0], p[1], ...
    # total pipeline combinations = pipeline_size ** dim_pipeline
    pipeline_total = 1
    for _ in range(dim_pipeline):
        pipeline_total *= pipeline_size

    ip_stride = pipeline_total
    t_stride = ip_range * pipeline_total

    for idx in prange(N):
        state = states_arr[idx]
        t = state[0]
        t_next = (t + 1) % n
        state_dim = state.shape[0]

        acts = actions_0 if period_flags[idx] == 1 else actions_t
        n_acts = acts.shape[0]

        best_cost = 1e10

        for a in range(n_acts):
            qr = acts[a, 0]
            qe = acts[a, 1]

            immediate_cost = qe * ce
            expected_cost = 0.0
            valid = True

            ip_new = state[1] + qe + state[2]  # ip + qe + pipeline[0]

            for d in range(demand_vals.shape[0]):
                dem = demand_vals[d]
                prob = demand_probs[d]
                ipe = int(ip_new) - dem

                # Build next state: (t_next, ipe, pipeline[1:], qr)
                # Compute flat index directly
                ip_next = ipe - ip_min
                if ip_next < 0 or ip_next >= ip_range:
                    valid = False
                    break

                # pipeline next = state[3:] + [qr]
                # compute pipeline flat index
                p_idx = 0
                stride = 1
                # last pipeline entry = qr
                if qr < 0 or qr > max_demand:
                    valid = False
                    break
                p_idx += qr * stride
                stride *= pipeline_size

                # remaining pipeline entries = state[3..state_dim-1] in reverse
                pipe_valid = True
                for k in range(state_dim - 3):
                    # state[state_dim-1-k] going backwards through pipeline[1:]
                    pval = state[state_dim - 1 - k]
                    if pval < 0 or pval > max_demand:
                        pipe_valid = False
                        break
                    p_idx += pval * stride
                    stride *= pipeline_size

                if not pipe_valid:
                    valid = False
                    break

                next_idx = t_next * t_stride + ip_next * ip_stride + p_idx

                if next_idx < 0 or next_idx >= N:
                    valid = False
                    break

                inv = ipe - state[2]  # ipe - pipeline[0]
                inv_cost = inv * h if inv >= 0 else -inv * b
                expected_cost += prob * (inv_cost + vf_arr[next_idx])

            if valid:
                total = immediate_cost + expected_cost
                if total < best_cost:
                    best_cost = total

        new_vf[idx] = best_cost

    return new_vf


@njit
def _vf_update_single(
    n: int,
    states_arr: np.ndarray,
    demand_vals: np.ndarray,
    demand_probs: np.ndarray,
    ce: float,
    h: float,
    b: float,
    idx: int,
    vf_arr: np.ndarray,
    actions: np.ndarray,
    n_periods: int,
    ip_min: int,
    ip_range: int,
    max_demand: int,
    dim_pipeline: int,
    N: int,
) -> Tuple[float, int, int]:
    """Serial single-state update for policy extraction."""
    state = states_arr[idx]
    t = state[0]
    t_next = (t + 1) % n
    state_dim = state.shape[0]

    pipeline_size = max_demand + 1
    pipeline_total = 1
    for _ in range(dim_pipeline):
        pipeline_total *= pipeline_size
    ip_stride = pipeline_total
    t_stride = ip_range * pipeline_total

    best_cost = 1e10
    best_qr = 0
    best_qe = 0

    for a in range(actions.shape[0]):
        qr = actions[a, 0]
        qe = actions[a, 1]

        immediate_cost = qe * ce
        expected_cost = 0.0
        valid = True

        ip_new = state[1] + qe + state[2]

        for d in range(demand_vals.shape[0]):
            dem = demand_vals[d]
            prob = demand_probs[d]
            ipe = int(ip_new) - dem

            ip_next = ipe - ip_min
            if ip_next < 0 or ip_next >= ip_range:
                valid = False
                break

            if qr < 0 or qr > max_demand:
                valid = False
                break

            p_idx = 0
            stride = 1
            p_idx += qr * stride
            stride *= pipeline_size

            pipe_valid = True
            for k in range(state_dim - 3):
                pval = state[state_dim - 1 - k]
                if pval < 0 or pval > max_demand:
                    pipe_valid = False
                    break
                p_idx += pval * stride
                stride *= pipeline_size

            if not pipe_valid:
                valid = False
                break

            next_idx = t_next * t_stride + ip_next * ip_stride + p_idx

            if next_idx < 0 or next_idx >= N:
                valid = False
                break

            inv = ipe - state[2]
            inv_cost = inv * h if inv >= 0 else -inv * b
            expected_cost += prob * (inv_cost + vf_arr[next_idx])

        if valid:
            total = immediate_cost + expected_cost
            if total < best_cost:
                best_cost = total
                best_qr = qr
                best_qe = qe

    return best_cost, best_qr, best_qe


class DynamicProgrammingParityController(BaseDualController):
    """
    Dynamic programming controller with a period-aware (parity) state.
    Optimized with numba prange for parallel state updates.

    Key optimization: replaces per-state Python loop + numba Dict lookup
    with a single parallel sweep using direct numpy array indexing.
    Results are numerically identical to the baseline.

    State layout : (t, ip, pipeline[0], ..., pipeline[dim-1])
    Action layout: (qr, qe) — qr > 0 only when t == 0
    """

    def __init__(self, cycle_length: int = 2) -> None:
        if cycle_length not in (1, 2, 3):
            raise ValueError("cycle_length must be 1, 2, or 3")
        self.cycle_length = cycle_length
        self.sourcing_model = None
        self.qf = None
        self.vf = None
        self._period: int = 0
        logger.info(
            f"Initialized DynamicProgrammingParityController (optimized) "
            f"with cycle_length={cycle_length}"
        )

    @staticmethod
    def _get_basestock_ub(
        exp_demand: float, lead_time: int, support: float, h: float, b: float
    ) -> float:
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
    ) -> None:
        self.sourcing_model = sourcing_model

        if not isinstance(sourcing_model.demand_generator, UniformDemand):
            raise ValueError(
                "DynamicProgrammingParityController only supports uniform demand."
            )
        if sourcing_model.expedited_lead_time != 0:
            raise ValueError(
                "DynamicProgrammingParityController only supports expedited_lead_time = 0."
            )

        start_time = datetime.now()
        logger.info(f"Starting optimized parity DP at {start_time}")

        min_demand = int(sourcing_model.demand_generator.get_min_demand())
        max_demand = int(sourcing_model.demand_generator.get_max_demand())
        exp_demand = (max_demand + min_demand) / 2.0
        support = max_demand - min_demand
        h = sourcing_model.get_holding_cost()
        b = sourcing_model.get_shortage_cost()
        ce = sourcing_model.get_expedited_order_cost()
        lr = sourcing_model.get_regular_lead_time()
        n = self.cycle_length

        base_e = self._get_basestock_ub(exp_demand, 0, support, h, b)
        base_r = self._get_basestock_ub(exp_demand, lr, support, h, b)
        min_ip = int(min(base_r, base_e) - max_demand)
        max_ip = int(max(n * base_r, base_e))
        max_order = max_ip + abs(min_ip)

        dim_pipeline = lr - 1  # le=0
        state_dim = 2 + dim_pipeline  # (t, ip, pipeline...)

        # ip range for indexing
        ip_min = -abs(min_ip)
        ip_max = max_ip
        ip_range = ip_max - ip_min + 1

        # Demand
        demand_vals = np.arange(min_demand, max_demand + 1, dtype=np.int64)
        demand_probs = np.full(
            len(demand_vals),
            1.0 / (max_demand - min_demand + 1),
            dtype=np.float64,
        )

        # Build state list
        states_list = list(
            product(
                range(n),
                range(ip_min, ip_max + 1),
                *(range(int(max_demand) + 1),) * int(dim_pipeline),
            )
        )
        N_states = len(states_list)
        logger.info(f"Total states: {N_states}, state_dim: {state_dim}")
        logger.info(f"ip_range: [{ip_min}, {ip_max}], max_order: {max_order}")

        states_arr = np.array(states_list, dtype=np.int64)
        period_flags = (states_arr[:, 0] == 0).astype(np.int64)

        # Build state_index map (Python dict, used only for qf extraction)
        state_index_py = {s: i for i, s in enumerate(states_list)}

        # Actions
        actions_0 = np.array(
            list(product(range(max_order), repeat=2)), dtype=np.int64
        )
        actions_t = np.array(
            [(0, qe) for qe in range(max_order)], dtype=np.int64
        )

        logger.info(
            f"Actions: period-0={len(actions_0)}, period-t={len(actions_t)}"
        )

        # VF array — flat, indexed by state position
        vf_arr = np.ones(N_states, dtype=np.float64)

        all_values = np.zeros(max_iterations, dtype=np.float64)
        val = 0.0

        logger.info("Starting value iteration (parallel sweep)...")
        for iteration in tqdm(range(max_iterations)):
            vf_arr = _vf_sweep_parallel(
                n, states_arr, period_flags,
                actions_0, actions_t,
                demand_vals, demand_probs,
                ce, h, b,
                vf_arr,
                n, ip_min, ip_range, max_demand, dim_pipeline,
            )

            this_average = np.mean(vf_arr[vf_arr < 1e9])
            val = this_average / (iteration + 1)
            all_values[iteration] = val

            if iteration > 1 and iteration % log_freq == 0:
                logger.info(f"Epoch {iteration} - Value: {val:.4f}")

            if iteration > 1 and iteration % validation_freq == 0:
                delta = abs(all_values[iteration - 1] - all_values[iteration])
                if delta <= tolerance:
                    logger.info(f"Converged at iteration {iteration}")
                    break

        # Extract policy
        logger.info("Extracting policy...")
        qf = {}
        for i, s in enumerate(states_list):
            acts = actions_0 if period_flags[i] == 1 else actions_t
            _, best_qr, best_qe = _vf_update_single(
                n, states_arr, demand_vals, demand_probs,
                ce, h, b,
                i, vf_arr, acts,
                n, ip_min, ip_range, max_demand, dim_pipeline,
                N_states,
            )
            qf[tuple(s)] = (int(best_qr), int(best_qe))

        self.qf = qf
        self.vf = val
        self._period = 0

        end_time = datetime.now()
        logger.info(f"Optimized parity DP completed in {end_time - start_time}")
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
    ) -> Union[Tuple[torch.Tensor, torch.Tensor], Tuple[int, int]]:
        if self.sourcing_model is None:
            raise AttributeError("The controller is not trained.")

        lr = self.sourcing_model.get_regular_lead_time()
        t = self._period

        current_inventory = self._check_current_inventory(current_inventory)
        past_regular_orders = self._check_past_orders(past_regular_orders, lr)

        first = (
            current_inventory.squeeze()
            + past_regular_orders.squeeze()[-lr]
        )
        second = past_regular_orders.squeeze()[-lr + 1:]
        key = tuple([t, int(first)] + second.int().tolist())

        action = self.qf[key]
        self._period = (self._period + 1) % self.cycle_length

        if output_tensor:
            return tuple(torch.tensor([[v]]) for v in action)
        return action

    def get_last_cost(self, sourcing_model: DualSourcingModel) -> torch.Tensor:
        last_regular_q = sourcing_model.get_last_regular_order()
        last_expedited_q = sourcing_model.get_last_expedited_order()
        regular_order_cost = sourcing_model.get_regular_order_cost()
        expedited_order_cost = sourcing_model.get_expedited_order_cost()
        holding_cost = sourcing_model.get_holding_cost()
        shortage_cost = sourcing_model.get_shortage_cost()
        current_inventory = sourcing_model.get_current_inventory()
        return (
            regular_order_cost * last_regular_q
            + expedited_order_cost * last_expedited_q
            + holding_cost * torch.relu(current_inventory)
            + shortage_cost * torch.relu(-current_inventory)
        )

    @no_type_check
    def get_total_cost(
        self,
        sourcing_model: DualSourcingModel,
        sourcing_periods: int,
        seed: Optional[int] = None,
    ) -> torch.Tensor:
        if seed is not None:
            torch.manual_seed(seed)

        self._period = 0
        total_cost = torch.tensor(0.0)

        for _ in tqdm(range(sourcing_periods)):
            for _ in range(self.cycle_length):
                current_inventory = sourcing_model.get_current_inventory()
                past_regular_orders = sourcing_model.get_past_regular_orders()
                past_expedited_orders = sourcing_model.get_past_expedited_orders()

                actions = self.predict(
                    current_inventory,
                    past_regular_orders,
                    past_expedited_orders,
                    output_tensor=True,
                )
                sourcing_model.order(actions[0], actions[1])
                total_cost += self.get_last_cost(sourcing_model).mean()

        return total_cost

    @no_type_check
    def get_average_cost(
        self,
        sourcing_model: DualSourcingModel,
        sourcing_periods: int,
        seed: Optional[int] = None,
    ) -> torch.Tensor:
        return (
            self.get_total_cost(sourcing_model, sourcing_periods, seed)
            / (sourcing_periods * self.cycle_length)
        )

    def reset(self) -> None:
        self.qf = None
        self.vf = None
        self.sourcing_model = None
        self._period = 0