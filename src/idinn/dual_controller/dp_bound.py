import logging
from datetime import datetime
from itertools import product
from typing import Dict as TypingDict
from typing import List as TypingList
from typing import Optional, Tuple, Union, no_type_check

import numpy as np
import torch
from numba import njit, types  # type: ignore
from numba.typed import Dict, List
from tqdm import tqdm

from ..demand import UniformDemand
from ..sourcing_model import DualSourcingModel
from .base import BaseDualController

# Get root logger
logger = logging.getLogger()


class DynamicProgrammingController(BaseDualController):
    def __init__(self, cycle_length: int = 2) -> None:
        if cycle_length not in (1, 2, 3):
            raise ValueError("cycle_length must be 1, 2, or 3")
        self.cycle_length = cycle_length
        self.sourcing_model = None
        self.qf = None
        self.vf = None
        logger.info(f"Initialized DynamicProgrammingController with cycle_length={cycle_length}")

    # ------------------------------------------------------------------
    # Bellman update functions — one per supported cycle length.
    # Action convention: (qr0, qe0, qe1, ..., qe_{N-1})
    #   qr0  — regular order placed once per cycle (period 0)
    #   qe_t — expedited order placed every period t in [0, N-1]
    #
    # Bound convention (single convention everywhere):
    #   min_ip is the SIGNED lower bound of the inventory-position grid
    #   (a negative number), max_ip the signed upper bound. Cycle-start
    #   states, mid-cycle positions, and vf keys all live in
    #   [min_ip, max_ip].
    # ------------------------------------------------------------------
    @staticmethod
    def _get_free_slot_indices(lr, cycle_length):
        """
        Return pipeline slot indices that can be non-zero at cycle-start states.
        Rule: state[k] can be non-zero iff k ≡ lr (mod cycle_length).
        """
        target = lr % cycle_length
        return [k for k in range(1, lr) if k % cycle_length == target]

    @staticmethod
    @njit
    def _vf_update_n1(
            demand_prob: TypingDict[int, float],
            min_demand: int,
            max_demand: int,
            ce: float,
            h: float,
            b: float,
            min_ip: int,
            max_ip: int,
            state: Tuple[int, ...],
            vf: TypingDict[Tuple[int, ...], float],
            actions: TypingList[Tuple[int, int]],
    ) -> Tuple[float, Optional[Tuple[int, int]]]:
        """
        Single-period Bellman update (N=1 cycle).

        Actions: (qr0, qe0).
        State: (ip, pipeline[0], ..., pipeline[dim-1])

        Note: min_ip and max_ip are accepted for dispatch consistency with
        _vf_update_n2 and _vf_update_n3 but are not used here, since N=1
        has no intermediate (mid-cycle) states; the vf membership test
        enforces the same grid.
        """
        best_action = None
        best_cost = 10e9

        for qr0, qe0 in actions:
            immediate_cost = qe0 * ce
            expected_cost = 0.0
            valid = True

            # PERIOD 0: qe0 arrives immediately (le=0), state[1] arrives, qr0 enters pipeline back
            ip0 = state[0] + qe0 + state[1]
            pipeline0 = state[2:]

            for dem0 in range(int(min_demand), int(max_demand) + 1):
                ipe0 = int(ip0) - dem0
                state_next = (ipe0,) + pipeline0 + (qr0,)

                # state_next is a cycle-start state — must be in vf and not
                # carry the "no valid action" sentinel
                if (state_next not in vf) or (vf[state_next] > 10e8):
                    valid = False
                    break

                inv0 = ipe0 - state[1]
                inv_cost0 = inv0 * h if inv0 >= 0 else -inv0 * b
                expected_cost += demand_prob[dem0] * (inv_cost0 + vf[state_next])

            if valid and (immediate_cost + expected_cost) < best_cost:
                best_cost = immediate_cost + expected_cost
                best_action = (qr0, qe0)

        return best_cost, best_action

    @staticmethod
    @njit
    def _vf_update_n2(
            demand_prob: TypingDict[int, float],
            min_demand: int,
            max_demand: int,
            ce: float,
            h: float,
            b: float,
            min_ip: int,
            max_ip: int,
            state: Tuple[int, ...],
            vf: TypingDict[Tuple[int, ...], float],
            actions: TypingList[Tuple[int, int, int]],
    ) -> Tuple[float, Optional[Tuple[int, int, int]]]:
        """
        Two-period cycle Bellman update (N=2).

        Period 0: place (qr0, qe0). Period 1: place (qe1) only.
        Actions: (qr0, qe0, qe1).
        State: (ip, pipeline[0], ..., pipeline[dim-1])

        Intermediate state (period 1 starting point) is checked via explicit,
        SIGNED IP bounds; only the final cycle-start state state_next is
        looked up in vf.
        """
        best_action = None
        best_cost = 10e9

        for qr0, qe0, qe1 in actions:
            immediate_cost = (qe0 + qe1) * ce
            expected_cost = 0.0
            valid = True

            for dem0 in range(int(min_demand), int(max_demand) + 1):
                if not valid:
                    break
                for dem1 in range(int(min_demand), int(max_demand) + 1):

                    # PERIOD 0: qe0 + state[1] arrive, qr0 enters pipeline back
                    ip0 = state[0] + qe0 + state[1]
                    pipeline0 = state[2:]
                    ipe0 = int(ip0) - dem0

                    # Mid-cycle bounds check — signed bounds, same box as vf grid
                    if ipe0 < min_ip or ipe0 > max_ip:
                        valid = False
                        break

                    inv0 = ipe0 - state[1]
                    inv_cost0 = inv0 * h if inv0 >= 0 else -inv0 * b

                    # state1 is the (mid-cycle) state at start of period 1
                    state1 = (ipe0,) + pipeline0 + (qr0,)

                    # PERIOD 1: qe1 + state1[1] arrive, 0 enters pipeline back
                    ip1 = state1[0] + qe1 + state1[1]
                    pipeline1 = state1[2:]
                    ipe1 = int(ip1) - dem1
                    state_next = (ipe1,) + pipeline1 + (0,)

                    # state_next IS a cycle-start state — must be in vf and not
                    # carry the "no valid action" sentinel
                    if (state_next not in vf) or (vf[state_next] > 10e8):
                        valid = False
                        break

                    inv1 = ipe1 - state1[1]
                    inv_cost1 = inv1 * h if inv1 >= 0 else -inv1 * b

                    prob = demand_prob[dem0] * demand_prob[dem1]
                    expected_cost += prob * (inv_cost0 + inv_cost1 + vf[state_next])

            if valid and (immediate_cost + expected_cost) < best_cost:
                best_cost = immediate_cost + expected_cost
                best_action = (qr0, qe0, qe1)

        return best_cost, best_action

    @staticmethod
    @njit
    def _vf_update_n3(
            demand_prob: TypingDict[int, float],
            min_demand: int,
            max_demand: int,
            ce: float,
            h: float,
            b: float,
            min_ip: int,
            max_ip: int,
            state: Tuple[int, ...],
            vf: TypingDict[Tuple[int, ...], float],
            actions: TypingList[Tuple[int, int, int, int]],
    ) -> Tuple[float, Optional[Tuple[int, int, int, int]]]:
        """
        Three-period cycle Bellman update (N=3).

        Period 0: place (qr0, qe0). Periods 1-2: place (qe_t) only.
        Actions: (qr0, qe0, qe1, qe2).
        State: (ip, pipeline[0], ..., pipeline[dim-1])

        Intermediate states (periods 1 and 2 starting points) are checked via
        explicit, SIGNED IP bounds; only the final cycle-start state
        state_next is looked up in vf.
        """
        best_action = None
        best_cost = 10e9

        for qr0, qe0, qe1, qe2 in actions:
            immediate_cost = (qe0 + qe1 + qe2) * ce
            expected_cost = 0.0
            valid = True

            for dem0 in range(int(min_demand), int(max_demand) + 1):
                if not valid:
                    break
                for dem1 in range(int(min_demand), int(max_demand) + 1):
                    if not valid:
                        break
                    for dem2 in range(int(min_demand), int(max_demand) + 1):

                        # PERIOD 0: qe0 + state[1] arrive, qr0 enters pipeline back
                        ip0 = state[0] + qe0 + state[1]
                        pipeline0 = state[2:]
                        ipe0 = int(ip0) - dem0

                        # Mid-cycle bounds check on ipe0 — signed bounds
                        if ipe0 < min_ip or ipe0 > max_ip:
                            valid = False
                            break

                        inv0 = ipe0 - state[1]
                        inv_cost0 = inv0 * h if inv0 >= 0 else -inv0 * b

                        # state1 is the (mid-cycle) state at start of period 1
                        state1 = (ipe0,) + pipeline0 + (qr0,)

                        # PERIOD 1: qe1 + state1[1] arrive, 0 enters pipeline back
                        ip1 = state1[0] + qe1 + state1[1]
                        pipeline1 = state1[2:]
                        ipe1 = int(ip1) - dem1

                        # Mid-cycle bounds check on ipe1 — signed bounds
                        if ipe1 < min_ip or ipe1 > max_ip:
                            valid = False
                            break

                        inv1 = ipe1 - state1[1]
                        inv_cost1 = inv1 * h if inv1 >= 0 else -inv1 * b

                        # state2 is the (mid-cycle) state at start of period 2
                        state2 = (ipe1,) + pipeline1 + (0,)

                        # PERIOD 2: qe2 + state2[1] arrive, 0 enters pipeline back
                        ip2 = state2[0] + qe2 + state2[1]
                        pipeline2 = state2[2:]
                        ipe2 = int(ip2) - dem2
                        state_next = (ipe2,) + pipeline2 + (0,)

                        # state_next IS a cycle-start state — must be in vf and
                        # not carry the "no valid action" sentinel
                        if (state_next not in vf) or (vf[state_next] > 10e8):
                            valid = False
                            break

                        inv2 = ipe2 - state2[1]
                        inv_cost2 = inv2 * h if inv2 >= 0 else -inv2 * b

                        prob = demand_prob[dem0] * demand_prob[dem1] * demand_prob[dem2]
                        expected_cost += prob * (inv_cost0 + inv_cost1 + inv_cost2 + vf[state_next])

            if valid and (immediate_cost + expected_cost) < best_cost:
                best_cost = immediate_cost + expected_cost
                best_action = (qr0, qe0, qe1, qe2)

        return best_cost, best_action

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
            Specifies how many iteration to run before checking the tolerance is reached,
            e.g. `validation_freq=10` runs validation every 10 epochs.
        log_freq : int, default is 100
            Specifies how many training epochs to run before logging the training loss.
        bound_slack : int, default is 1
            Multiplier for widening the IP box in truncation-sensitivity checks.
            Run once with 1 and once with 2; matching values certify that the
            truncation is non-binding.
        """
        self.sourcing_model = sourcing_model

        # Check demand is uniform distributed
        if not isinstance(sourcing_model.demand_generator, UniformDemand):
            raise ValueError(
                "DynamicProgrammingController only supports uniform demand distribution."
            )
        # Check if the expedited_lead_time is 0
        if sourcing_model.expedited_lead_time != 0:
            raise ValueError(
                "DynamicProgrammingController only supports expedited_lead_time = 0."
            )

        start_time = datetime.now()
        logger.info(f"Starting dynamic programming at {start_time}")
        logger.info(
            f"Sourcing model parameters: batch_size={self.sourcing_model.batch_size}, "
            f"lead_time={self.sourcing_model.lead_time}, init_inventory={self.sourcing_model.init_inventory.int().item()}, "
            f"demand_generator={self.sourcing_model.demand_generator.__class__.__name__}"
        )
        logger.info(
            f"Training parameters: max_iterations={max_iterations}, tolerance={tolerance}, "
            f"cycle_length={self.cycle_length}"
        )

        min_demand = int(sourcing_model.demand_generator.get_min_demand())
        max_demand = int(sourcing_model.demand_generator.get_max_demand())
        exp_demand = (max_demand + min_demand) / 2.0
        support = max_demand - min_demand
        h = sourcing_model.get_holding_cost()
        b = sourcing_model.get_shortage_cost()
        ce = sourcing_model.get_expedited_order_cost()
        le = sourcing_model.get_expedited_lead_time()
        lr = sourcing_model.get_regular_lead_time()

        base_e = DynamicProgrammingController._get_basestock_ub(
            exp_demand=exp_demand, lead_time=le, support=support, h=h, b=b
        )
        base_r = DynamicProgrammingController._get_basestock_ub(
            exp_demand=exp_demand, lead_time=lr, support=support, h=h, b=b
        )

        # Original single-period bounds — used ONLY to size the action grid,
        # under the original (width) convention. Unchanged.
        min_ip_single = int(min(base_r, base_e) - max_demand)
        max_ip_single = int(max(base_r, base_e))
        max_order_single = max_ip_single + min_ip_single

        # Cyclic scaling: regular order covers N periods, expedited stays per-period
        max_order_regular = int(self.cycle_length * max_order_single)
        max_order_expedited = max_order_single

        # ------------------------------------------------------------------
        # BOUND FIX: min_ip and max_ip are SIGNED bounds of the IP box, set
        # explicitly and decoupled from the base-stock arithmetic.
        #
        # Lower bound: room for uncorrectable within-cycle backlog. The worst
        # plausible dip is on the order of a cycle-plus-slack of maximum
        # demand with no arrivals, so we allow (n + 1) * D_max below zero.
        #
        # Upper bound: cycle coverage plus one period of slack, since
        # mid-cycle positions can exceed cycle-start positions when supply
        # bunches at the cycle boundary.
        #
        # bound_slack widens both sides for truncation-sensitivity re-runs.
        # ------------------------------------------------------------------
        min_ip = -bound_slack * (self.cycle_length + 1) * max_demand
        max_ip = int(max(base_r * self.cycle_length, base_e)) + bound_slack * max_demand

        # Pipeline must hold the largest regular order
        pipeline_ub = max(int(max_demand), max_order_regular)
        dim_pipeline = lr - le - 1

        logger.info(
            f"Bounds: max_order_regular={max_order_regular}, max_order_expedited={max_order_expedited}, "
            f"min_ip={min_ip}, max_ip={max_ip}, pipeline_ub={pipeline_ub}, dim_pipeline={dim_pipeline}"
        )

        demand_prob = Dict.empty(key_type=types.int64, value_type=types.float64)
        demand_prob_ = sourcing_model.demand_generator.enumerate_support()
        for k, v in demand_prob_.items():
            demand_prob[k] = v

        free_slots = self._get_free_slot_indices(lr, self.cycle_length)
        logger.info(f"Free pipeline slots: {free_slots} (of {list(range(1, lr))})")

        slot_ranges = [
            range(pipeline_ub + 1) if k in free_slots else [0]
            for k in range(1, lr)
        ]

        # BOUND FIX: min_ip is the signed bound itself — use it directly,
        # not range(-min_ip, ...). Cycle-start grid and mid-cycle checks now
        # share one box.
        states_ = list(product(
            range(min_ip, max_ip + 1),
            *slot_ranges,
        ))
        logger.info(f"Reachable state space: {len(states_):,}")

        states = List()
        for state in states_:
            states.append(state)

        # Actions: (qr0, qe0, qe1, ..., qe_{N-1})
        actions_ = list(product(
            range(max_order_regular+1),
            *([range(max_order_expedited+1)] * self.cycle_length),
        ))
        actions = List()
        for action in actions_:
            actions.append(action)

        logger.info(f"State space size: {len(states)}, Action space size: {len(actions)}")

        # Select Bellman update for the configured cycle length
        _vf_update = {
            1: DynamicProgrammingController._vf_update_n1,
            2: DynamicProgrammingController._vf_update_n2,
            3: DynamicProgrammingController._vf_update_n3,
        }[self.cycle_length]

        # Values can be initiated arbitrarily
        vals = np.repeat(1.0, len(states))
        vf_ = dict(zip(states, vals))
        vf = Dict.empty(
            key_type=types.UniTuple(types.int64, lr), value_type=types.float64
        )
        for k, v in vf_.items():
            vf[k] = v

        all_values = np.zeros(max_iterations, dtype=float)
        these_values = np.zeros(len(states))
        iteration_arr = []
        value_arr = []
        qf = {}
        val = 0

        for iteration in tqdm(range(max_iterations)):
            for idx, state in enumerate(states):
                these_values[idx] = _vf_update(
                    demand_prob, min_demand, max_demand, ce, h, b,
                    min_ip, max_ip,
                    state, vf, actions,
                )[0]
            for idx, state in enumerate(states):
                vf[state] = these_values[idx]

            iter_vals = np.array([val for val in vf.values() if val < 10e8])
            this_average = np.mean(iter_vals)

            val = this_average / (iteration + 1)
            all_values[iteration] = val

            if iteration > 1 and iteration % log_freq == 0:
                logger.info(
                    f"Epoch {iteration}/{max_iterations} - Value: {all_values[iteration]:.4f}"
                )

            if iteration > 1 and iteration % validation_freq == 0:
                iteration_arr.append(iteration)
                value_arr.append(all_values[iteration])
                delta = all_values[iteration - 1] - all_values[iteration]
                if delta <= tolerance:
                    logger.info(f"Converged at iteration {iteration}")
                    break
        else:
            logger.warning(
                f"Did not converge within max_iterations={max_iterations} "
                f"(tolerance={tolerance}). Extracting policy from current vf anyway."
            )

        # Policy extraction - always runs, converged or not (mirrors dp_bound_parallel.py)
        for state in states:
            qa = _vf_update(
                demand_prob, min_demand, max_demand, ce, h, b,
                min_ip, max_ip,
                state, vf, actions,
            )[1]
            if qa is not None:
                qf[state] = qa

        self.qf = qf
        self.vf = val

        # --- Post-fit diagnostics: warn if the action grid is saturated ---
        n_sat_r = sum(1 for a in qf.values() if a[0] == max_order_regular)
        n_sat_e = sum(
            1 for a in qf.values() if any(q == max_order_expedited for q in a[1:])
        )
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
            f"Final best cost: {self.get_average_cost(self.sourcing_model, sourcing_periods=1000, seed=42):.4f}"
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