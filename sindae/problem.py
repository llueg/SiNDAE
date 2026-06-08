"""
problem.py

ProblemDefinition: base class for Neural DAE problem specifications.

The user subclasses this and implements the abstract methods.
All other methods (including discretize) have sensible defaults.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List, Optional

import numpy as np
import pyomo.environ as pyo


class ProblemDefinition(ABC):
    """
    Base class for a Neural DAE problem.

    The user subclasses this and implements:
      build_trajectory(block, traj_idx)  — base DAE (no NN, no discretization)
      get_input_vars(block, t)           — raw Pyomo vars fed into the NN
      get_output_vars(block, t)          — raw Pyomo vars produced by the NN
      get_obs_vars(block, t)             — observed vars (default: same as get_input_vars)
      get_aux_vars(block, t)             — extra vars to track (default: none)

    Discretization uses Lagrange-Radau collocation by default; override
    ``discretize`` for custom schemes.

    Parameters
    ----------
    ics : np.ndarray
        Initial conditions, shape (num_trajectories, state_dim).
    input_dim : int
        Dimension of NN inputs (= len(get_input_vars(block, t))).
    z_dim : int
        Dimension of NN outputs (= len(get_output_vars(block, t))).
    t_span : tuple[float, float]
        Time horizon (t0, tf).
    nfe : int
        Number of finite elements for Radau collocation.
    ncp : int
        Number of collocation points per finite element.
    obs_times : List[np.ndarray], optional
        Observed time points per trajectory, shape (T_obs,) each.
    obs_values : List[np.ndarray], optional
        Observed values per trajectory, shape (T_obs, obs_dim) each.
    """

    def __init__(
        self,
        ics: np.ndarray,
        input_dim: int,
        z_dim: int,
        t_span: tuple,
        nfe: int,
        ncp: int,
        obs_times: Optional[List[np.ndarray]] = None,
        obs_values: Optional[List[np.ndarray]] = None,
        obs_dim: Optional[int] = None,
        aux_vars_dim: Optional[int] = None,
    ):
        ics = np.asarray(ics)
        num_trajectories = len(ics)
        if obs_times is not None:
            assert len(obs_times) == num_trajectories
        if obs_values is not None:
            assert len(obs_values) == num_trajectories
        self.ics = ics
        self.num_trajectories = num_trajectories
        self.input_dim = input_dim
        self.z_dim = z_dim
        self.t_span = t_span
        self.nfe = nfe
        self.ncp = ncp
        self.obs_times = obs_times
        self.obs_values = obs_values
        self.obs_dim: int = obs_dim if obs_dim is not None else input_dim
        self.aux_vars_dim: Optional[int] = aux_vars_dim
    # ------------------------------------------------------------------
    # Abstract interface — user must implement these
    # ------------------------------------------------------------------

    @abstractmethod
    def build_trajectory(self, block: pyo.Block, traj_idx: int) -> None:
        """Add base DAE variables and constraints to block (no NN, no discretization)."""

    @abstractmethod
    def get_input_vars(self, block: pyo.Block, t) -> list:
        """Return the list of raw Pyomo Var objects fed into the NN at time t."""

    @abstractmethod
    def get_output_vars(self, block: pyo.Block, t) -> list:
        """Return the list of raw Pyomo Var objects produced by the NN at time t."""

    # ------------------------------------------------------------------
    # Optional overrides — defaults are provided
    # ------------------------------------------------------------------

    def discretize(self, model: pyo.ConcreteModel) -> None:
        """
        Discretize all trajectories with Lagrange-Radau collocation.

        Override for non-standard schemes (e.g. different collocation type,
        per-trajectory nfe/ncp, or non-DAE problems).
        """
        for i in model.traj_set:
            pyo.TransformationFactory('dae.collocation').apply_to(
                model.trajectories[i],
                nfe=self.nfe, ncp=self.ncp,
                scheme='LAGRANGE-RADAU',
                wrt=model.trajectories[i].t,
            )

    def get_obs_vars(self, block: pyo.Block, t) -> list:
        """
        Return observed Pyomo vars used in the data-fit objective at time t.
        Default: same as get_input_vars.  Override when obs != NN inputs.
        """
        return self.get_input_vars(block, t)

    def get_aux_vars(self, block: pyo.Block, t) -> list:
        """
        Return additional Pyomo vars to track in InstanceData (e.g. algebraic vars).
        Default: none.  Override to populate TrajectoryData.aux_vars.
        """
        return []

    def add_true_output_constraints(self, block: pyo.Block) -> None:
        """
        Add constraints pinning the output vars to the true formula.

        Called pre-discretisation (block.t is still a ContinuousSet).
        Used only by generate_data — not part of normal training.
        """
        raise NotImplementedError(
            f"{type(self).__name__} does not define a true output formula. "
            "Implement add_true_output_constraints to use generate_data."
        )

    # ------------------------------------------------------------------
    # Observed-data statistics — computed from self.obs_values
    # ------------------------------------------------------------------

    @property
    def obs_mean(self) -> np.ndarray:
        """Mean of observed values across all trajectories (obs_dim,)."""
        if self.obs_values is None:
            raise RuntimeError("obs_values not set on problem")
        return np.mean(np.vstack(self.obs_values), axis=0)

    @property
    def obs_std(self) -> np.ndarray:
        """Std of observed values across all trajectories (obs_dim,), clipped to 1e-8."""
        if self.obs_values is None:
            raise RuntimeError("obs_values not set on problem")
        return np.std(np.vstack(self.obs_values), axis=0).clip(min=1e-8)
