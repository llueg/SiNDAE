"""
data_utils.py

Data containers for multi-trajectory Neural DAE solutions.

Classes
-------
TrajectoryData
    Per-trajectory arrays: sampling_times, nn_input, nn_output, obs, aux_vars.

InstanceData
    List of TrajectoryData with convenience list-accessor properties and
    normalization statistics (input_mean/std, output_mean/std) computed on the fly.

Functions
---------
extract_instance_data(problem, model) -> InstanceData
    Extract InstanceData from a solved Pyomo model using a ProblemDefinition.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional
import logging

import numpy as np
import pyomo.environ as pyo

from sindae.problem import ProblemDefinition
from sindae.solvers import make_nlp_solver

logger = logging.getLogger(__name__)

@dataclass
class TrajectoryData:
    sampling_times: np.ndarray           # (num_t,)
    nn_input:       np.ndarray           # (num_t, input_dim)
    nn_output:      np.ndarray           # (num_t, output_dim)
    obs:            np.ndarray           # (num_t, obs_dim) — model values of observed vars
    aux_vars:       Optional[np.ndarray] = None  # (num_t, aux_dim) or None


class InstanceData:
    """
    Container for multi-trajectory solution data extracted from a solved Pyomo model.

    Per-trajectory data lives in TrajectoryData objects (accessible via indexing).
    Normalization statistics (input_mean/std, output_mean/std) are computed on the
    fly from the stored nn_input / nn_output arrays across all trajectories.

    Note: obs is stored in original (un-normalised) space.  Obs normalisation
    statistics are available via problem.obs_mean / problem.obs_std.
    """

    def __init__(self, trajectories: List[TrajectoryData]) -> None:
        self._trajectories = trajectories

    def __getitem__(self, idx) -> TrajectoryData:
        return self._trajectories[idx]

    def __len__(self) -> int:
        return len(self._trajectories)
    
    def append_trajectory(self, traj_data: TrajectoryData) -> None:
        self._trajectories.append(traj_data)

    def save_to_file(self, filename: str) -> None:
        np.savez(filename, trajectories=self._trajectories)
    
    @staticmethod
    def load_from_file(filename: str) -> InstanceData:
        data = np.load(filename, allow_pickle=True)
        trajectories = data['trajectories'].tolist()
        return InstanceData(trajectories)

    @property
    def num_trajectories(self) -> int:
        return len(self._trajectories)

    # ── Per-trajectory list accessors ──────────────────────────────────────────

    @property
    def sampling_times(self) -> List[np.ndarray]:
        return [t.sampling_times for t in self._trajectories]

    @property
    def nn_input(self) -> List[np.ndarray]:
        return [t.nn_input for t in self._trajectories]

    @property
    def nn_output(self) -> List[np.ndarray]:
        return [t.nn_output for t in self._trajectories]

    @property
    def obs(self) -> List[np.ndarray]:
        return [t.obs for t in self._trajectories]

    # ── Normalization statistics (computed across all trajectories) ─────────────

    @property
    def input_mean(self) -> np.ndarray:
        return np.mean(np.vstack(self.nn_input), axis=0)

    @property
    def input_std(self) -> np.ndarray:
        return np.std(np.vstack(self.nn_input), axis=0).clip(min=1e-8)

    @property
    def output_mean(self) -> np.ndarray:
        return np.mean(np.vstack(self.nn_output), axis=0)

    @property
    def output_std(self) -> np.ndarray:
        return np.std(np.vstack(self.nn_output), axis=0).clip(min=1e-8)


def extract_instance_data(
    problem,
    model: pyo.ConcreteModel,
) -> InstanceData:
    """
    Extract trajectory data from a solved Pyomo model.

    Iterates over ``model.traj_set``, calling ``problem.get_input_vars``,
    ``get_output_vars``, ``get_obs_vars``, and (if non-empty)
    ``get_aux_vars`` at each time point.

    Parameters
    ----------
    problem : ProblemDefinition
    model   : pyo.ConcreteModel  (solved; must have model.traj_set / model.trajectories)

    Returns
    -------
    InstanceData
    """
    trajectories = []
    for i in model.traj_set:
        block    = model.trajectories[i]
        t_sorted = sorted(list(block.t))

        sampling_times = np.array(t_sorted)
        nn_input  = np.array(
            [[pyo.value(v) for v in problem.get_input_vars(block, t)] for t in t_sorted]
        )
        nn_output = np.array(
            [[pyo.value(v) for v in problem.get_output_vars(block, t)] for t in t_sorted]
        )
        obs = np.array(
            [[pyo.value(v) for v in problem.get_obs_vars(block, t)] for t in t_sorted]
        )

        aux_rows = [problem.get_aux_vars(block, t) for t in t_sorted]
        if aux_rows[0]:
            aux_vars = np.array([[pyo.value(v) for v in row] for row in aux_rows])
        else:
            aux_vars = None

        trajectories.append(TrajectoryData(sampling_times, nn_input, nn_output, obs, aux_vars))

    return InstanceData(trajectories)


def generate_data(
    problem: ProblemDefinition,
    obs_every: int = 1,
    seed: int = 0,
    noise_std: Optional[np.ndarray] = None,
    pounce_options: Optional[dict] = None,
    backend: str = 'pounce',
    tee: bool = False,
) -> InstanceData:
    """
    Solve the true model for all trajectories and populate problem with data.

    Builds a Pyomo model using problem.build_trajectory +
    problem.add_true_output_constraints + problem.discretize, solves with
    POUNCE, then:
      - Sets problem.obs_times  (subsampled collocation times)
      - Sets problem.obs_values (noisy observations of get_obs_vars)
      - Returns the true trajectories as an InstanceData

    Parameters
    ----------
    problem     : ProblemDefinition
        Must implement add_true_output_constraints.
    noise_std   : np.ndarray
        Std of Gaussian noise added to observations (0 = noiseless).
    obs_every   : int
        Keep every N-th collocation time point as an observation.
        1 = observe at all collocation points (default).
    seed        : int
        RNG seed for reproducible noise.
    pounce_options : dict, optional
        Extra NLP solver options, e.g. {'tol': 1e-9}.
    backend     : str  (default ``'pounce'``; ``'ipopt'`` / ``'cyipopt'``)
        NLP solver backend used for the true-model solve.
    tee         : bool
        Pass through to the NLP solver (print output if True).

    Returns
    -------
    InstanceData
        True trajectories at all collocation points (nn_input, nn_output, obs,
        and aux_vars per trajectory). Returns ``None`` if the POUNCE solve fails
        or does not reach optimality.
    """
    rng = np.random.default_rng(seed)
    obs_dim = problem.obs_dim
    if noise_std is None:
        noise_std = np.zeros((obs_dim,))
    else:
        noise_std = np.asarray(noise_std)
        assert noise_std.shape == (obs_dim,)

    # ── Build true model ──────────────────────────────────────────────────────
    m = pyo.ConcreteModel()
    m.traj_set     = pyo.RangeSet(0, problem.num_trajectories - 1)
    m.trajectories = pyo.Block(m.traj_set)

    for i in m.traj_set:
        block = m.trajectories[i]
        problem.build_trajectory(block, i)
        problem.add_true_output_constraints(block)

    problem.discretize(m)
    m.obj = pyo.Objective(expr=0.0)

    # ── Solve ─────────────────────────────────────────────────────────────────
    solver = make_nlp_solver(backend, pounce_options)
    try:
        result = solver.solve(m, tee=tee).result
    except Exception as e:
        logger.warning(f"generate_data: solve failed with error: {e}")
        return None
    logger.info(
        f"generate_data: {result.solver.status} / "
        f"{result.solver.termination_condition}"
    )
    if result.solver.termination_condition != pyo.TerminationCondition.optimal:
        logger.warning("generate_data: solve did not reach optimality; "
                       "results may be unreliable.")
        return None

    # ── Extract ───────────────────────────────────────────────────────────────
    obs_times_list   = []
    obs_values_list  = []

    for i in m.traj_set:
        block    = m.trajectories[i]
        t_sorted = sorted(list(block.t))
        obs_arr = np.array(
            [[pyo.value(v) for v in problem.get_obs_vars(block, t)]
             for t in t_sorted]
        )
        # Subsample
        idx     = np.arange(0, len(t_sorted), obs_every)
        t_obs   = np.array(t_sorted)[idx]
        obs_sub = obs_arr[idx].copy()

        # Add noise
        obs_sub = obs_sub + rng.normal(0.0, noise_std, obs_sub.shape)

        obs_times_list.append(t_obs)
        obs_values_list.append(obs_sub)

    # Populate problem with generated data
    problem.obs_times  = obs_times_list
    problem.obs_values = obs_values_list

    return extract_instance_data(problem, m)