"""
hybrid_dae.py

HybridDAE: the high-level fit/predict wrapper over the SiNDAE pipeline.

``fit`` orchestrates smoother -> pretrain -> train (simultaneous or
decomposition); ``predict`` embeds the trained network in a new problem and
solves the inference NLP.  Each stage is configured with its config dataclass
(:class:`SmootherConfig`, :class:`PretrainConfig`, :class:`SimultaneousConfig`
/ :class:`DecompConfig`, :class:`SolverConfig`), the same objects the
stage functions use, and everything is validated at construction so a typo
fails before any solve.  The stage functions (``solve_smoother``,
``pretrain_mlp``, ``solve_simultaneous``, ``train_decomp``,
``solve_inference``) remain the low-level escape hatch for power users, and
every intermediate the wrapper produces stays reachable as an attribute after
``fit``.

Example
-------
::

    import jax
    import numpy as np
    import sindae as sd

    problem = sd.LeslieGowerProblem(nfe=40, ncp=3)
    sd.generate_data(problem, noise_std=np.array([0.02, 0.02]), obs_every=4)

    mlp = sd.SimpleMLP(
        in_size=problem.input_dim, out_size=problem.z_dim,
        widths=[16, 16], activations=[jax.nn.softplus] * 2,
        key=jax.random.PRNGKey(0),
    )

    model = sd.HybridDAE(
        method="simultaneous",
        nlp_solver="pounce", linear_solver="feral",
        net=mlp,
        pretrain=sd.PretrainConfig(epochs=200),
        train=sd.SimultaneousConfig(reg_coef=1e-3),
        solver_options=sd.SolverConfig(tol=1e-6, max_iter=1000),
    )
    model.fit(problem)                  # smoother -> pretrain -> train

    # Inference on new initial conditions
    new_problem = sd.LeslieGowerProblem(ics=np.array([[1.2, 0.15]]),
                                        nfe=40, ncp=3)
    pred = model.predict(new_problem, slack_coef=1e-5)
    mu_hat = model.net                  # the trained SimpleMLP
"""
from __future__ import annotations

import warnings
from typing import Optional, Union

from sindae.data_utils import InstanceData, NormStats, extract_instance_data
from sindae.nn_utils import SimpleMLP
from sindae.problem import ProblemDefinition
from sindae.solvers import SolverConfig
from sindae.algorithms.smoother import SmootherConfig, solve_smoother
from sindae.algorithms.pretrain import PretrainConfig, pretrain_mlp
from sindae.algorithms.simultaneous.train import SimultaneousConfig, solve_simultaneous
from sindae.algorithms.decomp.train import DecompConfig, train_decomp
from sindae.algorithms.inference import solve_inference
from sindae.nn_utils import make_simple_mlp, _act_jax2str, _act_str2jax

import equinox as eqx
import jax
import json
import numpy as np
from jax2onnx import to_onnx

_METHODS = ("simultaneous", "decomposition")
_NLP_SOLVERS = ("pounce", "ipopt", "cyipopt")
_LINEAR_SOLVERS = ("feral", "ma27", "scipy")
_TRAIN_CONFIGS = {
    "simultaneous": SimultaneousConfig,
    "decomposition": DecompConfig,
}

def _require_config(value, cls, param: str) -> None:
    """Raise TypeError unless ``value`` is None or an instance of ``cls``."""
    if value is not None and not isinstance(value, cls):
        raise TypeError(
            f"{param}= takes a {cls.__name__} instance or None, "
            f"got {type(value).__name__}"
        )


class HybridDAE:
    """scikit-learn-style facade over the SiNDAE training pipeline.

    ``fit(problem)`` runs smoother -> pretrain -> train;
    ``predict(new_problem)`` runs inference with the trained network.

    Stage configuration comes in as the same config dataclasses the stage
    functions use; cross-cutting choices (``nlp_solver``, ``linear_solver``,
    ``solver_options``, ``unfix_io``) live only here, so no stage config can
    silently override them.

    Parameters
    ----------
    method : str
        Training approach: ``'simultaneous'`` (single NLP; default) or
        ``'decomposition'`` (Adam + KKT-gradient loop).
    nlp_solver : str
        NLP solver used at every stage: ``'pounce'`` (default), ``'ipopt'``,
        ``'cyipopt'``.
    linear_solver : str
        KKT/linear solver for the decomposition gradient back-solve:
        ``'feral'`` (default), ``'ma27'``, ``'scipy'``.  Unused by the
        simultaneous method.
    net : SimpleMLP
        The network to train, constructed outside the wrapper (see
        :class:`SimpleMLP`).  Its ``in_size`` / ``out_size`` must match
        ``problem.input_dim`` / ``problem.z_dim`` at ``fit``.
    smoother : SmootherConfig, optional
        Smoother-stage hyperparameters.  None uses ``SmootherConfig()``.
    pretrain : PretrainConfig, optional
        Supervised pretraining hyperparameters.  None uses
        ``PretrainConfig()`` (200 epochs); pass ``PretrainConfig(epochs=0)``
        to disable pretraining.
    train : SimultaneousConfig or DecompConfig, optional
        Training hyperparameters; the config class must match ``method``.
        None uses the method's config defaults.
    solver_options : SolverConfig, optional
        NLP solver options for the fit-time solves (smoother and training).
        The inference solve in ``predict`` does not inherit these; pass
        ``predict(..., solver_options=...)`` to tune it.
    unfix_io : bool
        Unfix the NN input/output variables in the smoother and training
        models (default True).  Set False for partially observed problems:
        unmeasured states have no data anchor, and leaving their variables
        free makes the solves diverge.

    Attributes
    ----------
    net : SimpleMLP
        The trained network (available after ``fit``).
    termination : str or None
        Termination condition of the training solve (e.g. ``'optimal'``).
        None for the decomposition method, whose per-step inner solves are
        tracked in ``history`` instead.  A non-optimal termination also
        raises a ``UserWarning`` during ``fit``.
    smoother_model : pyo.ConcreteModel
        The solved smoother NLP.
    smoother_data : InstanceData
        Extracted from the smoother; supplies the normalization statistics
        used by training and, later, ``predict``.
    training_model : pyo.ConcreteModel
        The solved training NLP (simultaneous) or the final decomposition
        iterate's model.
    trained_data : InstanceData
        Extracted from ``training_model``.
    history : dict or None
        Decomposition training history (``obj_history``, ...); None for the
        simultaneous method.
    inference_model : pyo.ConcreteModel
        The most recent solved inference NLP (set by ``predict``).
    """

    def __init__(
        self,
        method: str = "simultaneous",
        nlp_solver: str = "pounce",
        linear_solver: str = "feral",
        net: Optional[SimpleMLP] = None,
        smoother: Optional[SmootherConfig] = None,
        pretrain: Optional[PretrainConfig] = None,
        train: Union[SimultaneousConfig, DecompConfig, None] = None,
        solver_options: Optional[SolverConfig] = None,
        unfix_io: bool = True,
    ):
        if method not in _METHODS:
            raise ValueError(
                f"Unknown method {method!r}; choose from {sorted(_METHODS)}"
            )
        if isinstance(nlp_solver, str) and nlp_solver.lower() not in _NLP_SOLVERS:
            raise ValueError(
                f"Unknown NLP solver {nlp_solver!r}; choose from "
                f"{sorted(_NLP_SOLVERS)}"
            )
        if (isinstance(linear_solver, str)
                and linear_solver.lower() not in _LINEAR_SOLVERS):
            raise ValueError(
                f"Unknown linear solver {linear_solver!r}; choose from "
                f"{sorted(_LINEAR_SOLVERS)}"
            )

        if not isinstance(net, SimpleMLP):
            raise TypeError(
                f"net= takes a SimpleMLP instance (construct the network "
                f"outside the wrapper), got {type(net).__name__}"
            )

        _require_config(smoother, SmootherConfig, "smoother")
        _require_config(pretrain, PretrainConfig, "pretrain")
        _require_config(solver_options, SolverConfig, "solver_options")
        expected_train = _TRAIN_CONFIGS[method]
        if train is not None:
            if not isinstance(train, tuple(_TRAIN_CONFIGS.values())):
                raise TypeError(
                    f"train= takes a {expected_train.__name__} instance or "
                    f"None, got {type(train).__name__}"
                )
            if not isinstance(train, expected_train):
                raise ValueError(
                    f"train config {type(train).__name__} does not match "
                    f"method {method!r}; expected {expected_train.__name__}"
                )
        self.method = method
        self.nlp_solver = nlp_solver
        self.linear_solver = linear_solver
        self._net_init = net
        self.smoother = smoother
        self.pretrain = pretrain
        self.train = train
        self.solver_options = solver_options
        self.unfix_io = unfix_io

        self._net: Optional[SimpleMLP] = None
        self.termination: Optional[str] = None
        self.smoother_model = None
        self.smoother_data: Optional[InstanceData] = None
        self.training_model = None
        self.trained_data: Optional[InstanceData] = None
        self.history: Optional[dict] = None
        self.inference_model = None

    # ------------------------------------------------------------------

    @property
    def net(self) -> SimpleMLP:
        """The trained network.  Available after ``fit``."""
        self._check_fitted()
        return self._net

    def _check_fitted(self) -> None:
        if self._net is None:
            raise RuntimeError(
                "This HybridDAE is not fitted yet; call fit(problem) first."
            )

    @staticmethod
    def _check_solve(stage: str, model) -> str:
        """Return the solve's termination condition, warning if not optimal."""
        tc = str(model._solver_result.solver.termination_condition)
        if tc != "optimal":
            warnings.warn(
                f"HybridDAE {stage} solve terminated with {tc!r} (not "
                f"optimal); results may be unreliable.",
                stacklevel=3,
            )
        return tc

    # ------------------------------------------------------------------

    def fit(self, problem: ProblemDefinition, tee: bool = False) -> "HybridDAE":
        """Run the training pipeline on ``problem`` and return ``self``.

        Stages: solve the smoother, extract normalization data, pretrain the
        network on the smoother arrays, then train with the configured method
        (simultaneous NLP or decomposition loop).  Any
        non-optimal solve raises a ``UserWarning``; the training solve's
        termination condition lands on ``self.termination``.

        Parameters
        ----------
        problem : ProblemDefinition
            Must carry observations (``obs_times`` / ``obs_values``), set
            directly or via :func:`generate_data`.
        tee : bool
            Stream solver output to stdout (simultaneous method only).

        Returns
        -------
        self : HybridDAE
            Fitted wrapper; the trained network is ``self.net``.
        """
        mlp = self._net_init
        if (mlp.in_size, mlp.out_size) != (problem.input_dim, problem.z_dim):
            raise ValueError(
                f"net does not match the problem: (in_size, out_size) = "
                f"({mlp.in_size}, {mlp.out_size}), but the problem needs "
                f"({problem.input_dim}, {problem.z_dim})"
            )
        if problem.obs_times is None or problem.obs_values is None:
            raise ValueError(
                "problem has no observation data: set problem.obs_times / "
                "problem.obs_values (or call generate_data) before fit()"
            )

        smoother_cfg = self.smoother if self.smoother is not None else SmootherConfig()

        smoother_model = solve_smoother(
            problem, mlp,
            smooth_coef=smoother_cfg.smooth_coef,
            solver_options=self.solver_options,
            nlp_solver=self.nlp_solver,
            unfix_io=self.unfix_io,
        )
        self._check_solve("smoother", smoother_model)
        smoother_data = extract_instance_data(problem, smoother_model)

        pretrain_cfg = self.pretrain if self.pretrain is not None else PretrainConfig()
        mlp = pretrain_mlp(mlp, smoother_data, pretrain_cfg)

        if self.method == "simultaneous":
            cfg = self.train if self.train is not None else SimultaneousConfig()
            training_model, trained_net = solve_simultaneous(
                problem, mlp, cfg, data=smoother_data,
                smoother_model=smoother_model,
                solver_options=self.solver_options,
                nlp_solver=self.nlp_solver,
                tee=tee,
                unfix_io=self.unfix_io,
            )
            history = None
            termination = self._check_solve("training", training_model)
        else:
            cfg = self.train if self.train is not None else DecompConfig()
            training_model, trained_net, history = train_decomp(
                problem, mlp, cfg, data=smoother_data,
                smoother_model=smoother_model,
                solver_options=self.solver_options,
                nlp_solver=self.nlp_solver,
                linear_solver=self.linear_solver,
                unfix_io=self.unfix_io,
            )
            termination = None  # per-step inner solves; see history

        self._net = trained_net
        self.termination = termination
        self.smoother_model = smoother_model
        self.smoother_data = smoother_data
        self.training_model = training_model
        self.trained_data = extract_instance_data(problem, training_model)
        self.history = history
        return self

    def predict(
        self,
        problem: ProblemDefinition,
        slack_coef: float = 0.0,
        solver_options: Optional[SolverConfig] = None,
        tee: bool = False,
    ) -> InstanceData:
        """Embed the trained network in ``problem`` and solve the inference NLP.

        Normalization statistics are the ones the training stage consumed
        (``self.smoother_data``), so the network is evaluated in the space it
        was trained in.  The solved model is kept on ``self.inference_model``;
        a non-optimal solve raises a ``UserWarning``.

        Parameters
        ----------
        problem : ProblemDefinition
            The problem to predict, e.g. the training system with new initial
            conditions.  Observations are not required.
        slack_coef : float
            0 (default) enforces the NN equality as a hard constraint; > 0
            relaxes it with l1 slack variables (see :func:`solve_inference`).
        solver_options : SolverConfig, optional
            NLP solver options for this inference solve.  Defaults to the
            solver's own defaults (independent of the constructor's fit-time
            ``solver_options``), so a bare ``predict`` matches a bare
            :func:`solve_inference` call.
        tee : bool
            Stream solver output to stdout.

        Returns
        -------
        InstanceData
            Predicted trajectories at the collocation points.
        """
        self._check_fitted()
        _require_config(solver_options, SolverConfig, "solver_options")
        if (problem.input_dim, problem.z_dim) != (self._net.in_size,
                                                  self._net.out_size):
            raise ValueError(
                f"problem does not match the trained net: problem needs "
                f"(in_size, out_size) = ({problem.input_dim}, {problem.z_dim}) "
                f"but the net has ({self._net.in_size}, {self._net.out_size})"
            )

        m = solve_inference(
            problem, self._net, self.smoother_data,
            slack_coef=slack_coef,
            solver_options=solver_options,
            nlp_solver=self.nlp_solver,
            tee=tee,
        )
        self._check_solve("inference", m)
        self.inference_model = m
        return extract_instance_data(problem, m)


    def save(self, path) -> None:
        """Serialize the trained network and its scaler to ``path``.

        Writes a one-line JSON manifest (architecture, activation names, the
        four normalization vectors from ``smoother_data``, plus ``method`` and
        ``termination``) followed by the Equinox leaf arrays.  Reload with
        :meth:`HybridDAE.load`.

        Only the network and scaler are persisted, not the stage configs or the
        training trajectories, so a loaded model can ``predict`` or warm-start a
        fresh ``fit`` but cannot reproduce the original solve bit-for-bit.

        Parameters
        ----------
        path : str or os.PathLike
            Destination file (the parent directory must exist).
        """
        self._check_fitted()
        sd = self.smoother_data
        manifest = {
            "format_version": 1,
            "in_size": self._net.in_size,
            "out_size": self._net.out_size,
            "widths": list(self._net.widths),
            "activations": [_act_jax2str(a) for a in self._net.activations],
            "norm": {
                "input_mean":  np.asarray(sd.input_mean).tolist(),
                "input_std":   np.asarray(sd.input_std).tolist(),
                "output_mean": np.asarray(sd.output_mean).tolist(),
                "output_std":  np.asarray(sd.output_std).tolist(),
            },
            "method": self.method,
            "termination": self.termination,
        }
        with open(path, "wb") as f:
            f.write((json.dumps(manifest) + "\n").encode())
            eqx.tree_serialise_leaves(f, self._net)

    @classmethod
    def load(cls, path) -> "HybridDAE":
        """Reconstruct a fitted :class:`HybridDAE` from a :meth:`save` file.

        The returned wrapper can ``predict`` immediately (the scaler is
        restored on ``smoother_data`` as a :class:`NormStats`) or ``fit`` again
        to warm-start from the loaded weights.  ``trained_data`` and the stage
        configs are not restored (they are not persisted); ``fit`` would use the
        default configs.

        Parameters
        ----------
        path : str or os.PathLike
            A file written by :meth:`save`.

        Returns
        -------
        HybridDAE
            A fitted wrapper carrying the loaded network and scaler.
        """
        with open(path, "rb") as f:
            manifest = json.loads(f.readline().decode())
            skeleton = make_simple_mlp(
                key=jax.random.PRNGKey(0),
                in_size=manifest["in_size"],
                out_size=manifest["out_size"],
                widths=manifest["widths"],
                activations=[_act_str2jax(s) for s in manifest["activations"]],
            )
            net = eqx.tree_deserialise_leaves(f, skeleton)

        model = cls(net=net, method=manifest["method"])
        model._net = net
        norm = manifest["norm"]
        model.smoother_data = NormStats(
            input_mean=np.asarray(norm["input_mean"]),
            input_std=np.asarray(norm["input_std"]),
            output_mean=np.asarray(norm["output_mean"]),
            output_std=np.asarray(norm["output_std"]),
        )
        model.termination = manifest["termination"]
        return model


    # TODO: Function to export the NN weights for another modelling software like OMLT (ONNX, or JSON format)
    def export(self, path="exported_models/", format="ONNX"):
        """
        Export the trained NN into ONNX, or json format with the scaler and IO 
        contract based on the model pyomo model.
        """

        self._check_fitted()

        if format=="json":
            # Fill in with json saving logic
            pass
        else:
            to_onnx(
                self.net, 
                [("B", 32)], 
                enable_double_precision=True,
                return_mode="file", 
                output_path="model.onnx"
                )




