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
from tabulate import tabulate

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


def _capture_io_names(problem: "ProblemDefinition", model) -> dict:
    """Record the ordered NN input/output variable names from a built model.

    Reads the first trajectory block's variables at its first time point, so the
    export IO contract survives even after the model is gone (it is persisted by
    :meth:`HybridDAE.save`).
    """
    block0 = model.trajectories[0]
    t0 = min(block0.t)
    return {
        "inputs": _slot_names(problem.get_input_vars(block0, t0)),
        "outputs": _slot_names(problem.get_output_vars(block0, t0)),
    }


def _slot_names(vars_at_t) -> list:
    """Human-readable per-slot names for a list of Pyomo Var elements.

    Used to record the export IO contract: the ordered physical meaning of each
    NN input/output slot, which no NN interchange format carries.  When several
    slots come from the same component (e.g. ``x[t, 0]`` and ``x[t, 1]`` both
    belong to ``x``), they are disambiguated positionally (``x[0]``, ``x[1]``);
    a component contributing a single slot keeps its bare name.
    """
    comps = [v.parent_component().local_name for v in vars_at_t]
    totals: dict = {}
    for c in comps:
        totals[c] = totals.get(c, 0) + 1
    seen: dict = {}
    names = []
    for c in comps:
        if totals[c] > 1:
            j = seen.get(c, 0)
            names.append(f"{c}[{j}]")
            seen[c] = j + 1
        else:
            names.append(c)
    return names


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
        self.io_names: Optional[dict] = None

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

    def fit(self, problem: ProblemDefinition, 
            metrics: Optional[list[str]] = None, 
            tee: bool = False) -> "HybridDAE":
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
        metrics : Optional[list[str]]
            List of metrics to be printed after training. Comparisons per
            state variable, per trajectory. Options: `mse`, `rmse`, `mae`.
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
        
        if metrics:
            for metric in metrics:
                if metric not in _METRICS.keys():
                    raise ValueError(
                        f'metric: "{metric}" not available '
                        f"please choose a metric from: {_METRICS.keys()}"
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
        io_names = _capture_io_names(problem, smoother_model)

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
        self.io_names = io_names

        if metrics:
            x, x_hat = _filter_data_from_collocation_points(problem, self.trained_data)
            row_labels = np.asarray([[f"traj_{i}"] for i in range(problem.num_trajectories)])
            print("=== Per Trajectory Metrics ===")
            for metric in metrics:
                print(f"{metric.upper()}: ")    
                metric_table = np.hstack((row_labels, _METRICS[metric](x, x_hat)))
                print(tabulate(
                    tabular_data=metric_table,
                    headers=[f"x_{i}" for i in range(problem.obs_dim)],
                    tablefmt="fancy_grid"
                ))

        return self

    def predict(
        self,
        problem: ProblemDefinition,
        slack_coef: float = 0.0,
        solver_options: Optional[SolverConfig] = None,
        eval_metrics: Optional[list[str]] = None,
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
            conditions.  Observations are not required unless ``eval_metrics``
            is set.
        slack_coef : float
            0 (default) enforces the NN equality as a hard constraint; > 0
            relaxes it with l1 slack variables (see :func:`solve_inference`).
        solver_options : SolverConfig, optional
            NLP solver options for this inference solve.  Defaults to the
            solver's own defaults (independent of the constructor's fit-time
            ``solver_options``), so a bare ``predict`` matches a bare
            :func:`solve_inference` call.
        eval_metrics : Optional[list[str]]
            Metrics to print, comparing the prediction against ``problem``'s
            observations per state variable and trajectory.  Options: ``mse``,
            ``rmse``, ``mae``.  Requires ``problem`` to carry observations.
        tee : bool
            Stream solver output to stdout.

        Returns
        -------
        InstanceData
            Predicted trajectories at the collocation points.
        """
        self._check_fitted()
        _require_config(solver_options, SolverConfig, "solver_options")

        if not problem.obs_values and not problem.obs_times and eval_metrics:
            raise ValueError(
                "evaluation metrics not available without `obs_values` or 'obs_times` "
                "defined for `problem: ProblemDefinition` arg"
            )

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

        inference_data = extract_instance_data(problem, m)

        if eval_metrics:
            x, x_hat = _filter_data_from_collocation_points(problem, inference_data)
            row_labels = np.asarray([[f"traj_{i}"] for i in range(problem.num_trajectories)])
            print("=== Per Trajectory Metrics ===")
            for metric in eval_metrics:
                print(f"{metric.upper()}: ")    
                metric_table = np.hstack((row_labels, _METRICS[metric](x, x_hat)))
                print(tabulate(
                    tabular_data=metric_table,
                    headers=[f"x_{i}" for i in range(problem.obs_dim)],
                    tablefmt="fancy_grid"
                ))

        return inference_data


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
            "io_names": self.io_names,
        }
        with open(path, "wb") as f:
            f.write((json.dumps(manifest) + "\n").encode())
            eqx.tree_serialise_leaves(f, self._net)

    @classmethod
    def load(cls, path: str, verbose: bool =False) -> "HybridDAE":
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
        verbose : bool
            Prints the loaded model information contained in the manifest.

        Returns
        -------
        HybridDAE
            A fitted wrapper carrying the loaded network and scaler.
        """
        with open(path, "rb") as f:
            manifest = json.loads(f.readline().decode())

            if verbose:
                print("Loaded model information:")
                for k, v in manifest.items():
                    if k == "norm":
                        print(f"  {k}:")
                        for nk, nv in v.items():
                            print(f"    {nk}: {np.asarray(nv)}")
                    else:
                        print(f"  {k}: {v}")

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
        model.io_names = manifest.get("io_names")
        return model


    def _export_bundle(self) -> dict:
        """Assemble the format-agnostic export payload from the fitted model.

        Everything a foreign optimization tool needs to embed the trained
        surrogate: the layer weights/biases, the hidden-layer activation names,
        the scaler (four normalization vectors), data-derived input bounds (the
        surrogate's trust region), and the ordered IO variable-name contract.
        """
        self._check_fitted()
        net = self._net
        sd = self.smoother_data 
        input_bounds = None
        if hasattr(sd, "nn_input"):  # InstanceData (not the loaded NormStats)
            stacked = np.vstack(sd.nn_input)
            input_bounds = [
                (float(lo), float(hi))
                for lo, hi in zip(stacked.min(axis=0), stacked.max(axis=0))
            ]
        return {
            "in_size": net.in_size,
            "out_size": net.out_size,
            "widths": list(net.widths),
            "weights": [np.asarray(layer.weight) for layer in net.layers],
            "biases": [np.asarray(layer.bias) for layer in net.layers],
            "activations": [_act_jax2str(a) for a in net.activations],
            "scaler": {
                "input_mean":  np.asarray(sd.input_mean),
                "input_std":   np.asarray(sd.input_std),
                "output_mean": np.asarray(sd.output_mean),
                "output_std":  np.asarray(sd.output_std),
            },
            "input_bounds": input_bounds,
            "io_names": self.io_names,
        }

    def export(self, path=None, format: Optional[str] = None,
               scaled: bool = False) -> str:
        """Export the trained network to a file for a foreign optimization tool.

        Unlike :meth:`save` (which round-trips back into SiNDAE), ``export`` is a
        one-way handoff.  Two file targets, both carrying the scaler so the
        network is evaluated in the space it was trained in:

        * ``'onnx'`` — writes the network graph to ``path`` and a ``<path>.json``
          sidecar with the scaler, input bounds, and IO contract.  By default
          (``scaled=False``) the graph is in normalized space and the scaler is
          kept out of it, because every OMLT loader applies the scaler
          separately.  With ``scaled=True`` the four normalization vectors are
          baked into the graph as affine layers, so the exported model consumes
          raw physical inputs and returns raw physical outputs (self-contained
          inference in any ONNX runtime, no sidecar arithmetic).  Needs the
          ``onnx`` extra.
        * ``'json'`` — writes the whole bundle (weights, activations, scaler,
          bounds, IO contract) as plain text. Only use for very small MLPs since 
          storing many weights may lead to large file sizes.

        For an in-memory OMLT model (not a file), use :meth:`to_omlt`.

        Parameters
        ----------
        path : str or os.PathLike
            Output file.  When ``format`` is omitted the target is inferred from
            the suffix (``.onnx`` / ``.json``).
        format : str, optional
            ``'onnx'`` or ``'json'``.  Required when ``path`` has no recognized
            suffix.
        scaled : bool, default False
            ONNX only.  When ``True``, bake the scaler into the exported graph
            so it maps raw physical inputs to raw physical outputs.  Rejected for
            ``'json'`` export (whose bundle always carries the scaler verbatim).

        Returns
        -------
        str
            The written path.
        """
        self._check_fitted()
        fmt = format.lower() if format is not None else None
        if fmt is None and path is not None:
            suffix = str(path).rsplit(".", 1)[-1].lower() if "." in str(path) else ""
            fmt = {"onnx": "onnx", "json": "json"}.get(suffix)
        if fmt not in ("onnx", "json"):
            raise ValueError(
                "export needs format='onnx'|'json' (or a path ending in "
                f"'.onnx'/'.json'); got format={format!r}, path={path!r}. "
            )
        if scaled and fmt != "onnx":
            raise ValueError(
                "scaled=True is only meaningful for ONNX export; the json "
                "bundle already carries the scaler verbatim."
            )

        bundle = self._export_bundle()
        if fmt == "onnx":
            return _export_onnx(bundle, self._net, path, scaled=scaled)
        return _export_json(bundle, path)

    def to_omlt(self):
        """Build an in-memory OMLT model of the trained network.

        Returns an ``omlt.neuralnet.NetworkDefinition`` with the normalization
        attached as an ``OffsetScaling`` (so the OMLT block's inputs/outputs are
        the raw physical variables, not normalized ones) and the data-derived
        input bounds as its ``scaled_input_bounds``.  Feed it to an OMLT
        formulation (e.g. ``FullSpaceSmoothNNFormulation``) inside your own
        optimization model.  Needs the ``omlt`` extra.

        Returns
        -------
        omlt.neuralnet.NetworkDefinition
        """
        return _export_omlt(self._export_bundle())


# --------------------------------------------------------------------------- #
# Export writers (one bundle -> N encodings).  OMLT / ONNX imports are lazy so
# the core package installs and imports without those optional dependencies.
# --------------------------------------------------------------------------- #

def _scaler_to_lists(scaler: dict) -> dict:
    return {k: np.asarray(v).tolist() for k, v in scaler.items()}


def _export_json(bundle: dict, path) -> str:
    """Write the full bundle as plain-text JSON."""
    if path is None:
        raise ValueError("json export needs a path")
    payload = {
        "format": "sindae-export",
        "format_version": 1,
        "in_size": bundle["in_size"],
        "out_size": bundle["out_size"],
        "widths": bundle["widths"],
        "activations": bundle["activations"],
        "weights": [np.asarray(w).tolist() for w in bundle["weights"]],
        "biases": [np.asarray(b).tolist() for b in bundle["biases"]],
        "scaler": _scaler_to_lists(bundle["scaler"]),
        "input_bounds": bundle["input_bounds"],
        "io_names": bundle["io_names"],
    }
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)
    return path


def _export_omlt(bundle: dict):
    """Build an OMLT NetworkDefinition with the scaler as an OffsetScaling."""
    try:
        from omlt.neuralnet import NetworkDefinition
        from omlt.neuralnet.layer import DenseLayer, InputLayer
        from omlt.scaling import OffsetScaling
    except ImportError as e:  # pragma: no cover - exercised only without omlt
        raise ImportError(
            "OMLT export needs the optional 'omlt' extra: pip install 'sindae[omlt]'"
        ) from e

    sc = bundle["scaler"]
    input_mean = np.asarray(sc["input_mean"])
    input_std = np.asarray(sc["input_std"])
    scaler = OffsetScaling(
        offset_inputs=input_mean.tolist(),
        factor_inputs=input_std.tolist(),
        offset_outputs=np.asarray(sc["output_mean"]).tolist(),
        factor_outputs=np.asarray(sc["output_std"]).tolist(),
    )

    scaled_bounds = None
    if bundle["input_bounds"] is not None:
        scaled_bounds = {
            i: (float((lo - input_mean[i]) / input_std[i]),
                float((hi - input_mean[i]) / input_std[i]))
            for i, (lo, hi) in enumerate(bundle["input_bounds"])
        }

    net_def = NetworkDefinition(scaling_object=scaler,
                                scaled_input_bounds=scaled_bounds)
    input_layer = InputLayer([bundle["in_size"]])
    net_def.add_layer(input_layer)

    sizes = [bundle["in_size"], *bundle["widths"], bundle["out_size"]]
    # Hidden layers carry their activation; the output layer is linear (None).
    activations = [*bundle["activations"], None]
    prev = input_layer
    for k, (w, b, act) in enumerate(
        zip(bundle["weights"], bundle["biases"], activations)
    ):
        # Equinox Linear stores (out, in); OMLT DenseLayer wants (in, out).
        dense = DenseLayer(
            [sizes[k]], [sizes[k + 1]],
            weights=np.asarray(w).T, biases=np.asarray(b), activation=act,
        )
        net_def.add_layer(dense)
        net_def.add_edge(prev, dense)
        prev = dense
    return net_def


class _ScaledMLP(eqx.Module):
    """Wraps a SimpleMLP with its scaler so the graph is raw-in / raw-out.

    Applies the same affine maps the Pyomo model embeds around the network
    (``x_norm = (x - input_mean) / input_std`` on the way in,
    ``y = output_mean + output_std * y_norm`` on the way out), so exporting this
    module bakes the scaler into the ONNX graph as affine ops.
    """

    net: SimpleMLP
    input_mean: jax.Array
    input_std: jax.Array
    output_mean: jax.Array
    output_std: jax.Array

    def __init__(self, net: SimpleMLP, scaler: dict):
        import jax.numpy as jnp

        self.net = net
        self.input_mean = jnp.asarray(scaler["input_mean"])
        self.input_std = jnp.asarray(scaler["input_std"])
        self.output_mean = jnp.asarray(scaler["output_mean"])
        self.output_std = jnp.asarray(scaler["output_std"])

    def __call__(self, x):  # x is ONE raw physical sample, like SimpleMLP
        x = (x - self.input_mean) / self.input_std
        y = self.net(x)
        return self.output_mean + self.output_std * y


def _export_onnx(bundle: dict, net: SimpleMLP, path, scaled: bool = False) -> str:
    """Write the network graph plus a scaler sidecar JSON.

    ``scaled=False`` writes the bare network (normalized space); ``scaled=True``
    included the scaler into the graph (raw physical in/out).  The sidecar records
    which contract applies via the ``"scaling"`` field so a consumer never
    double-applies the scaler.
    """
    if path is None:
        raise ValueError("onnx export needs a path")
    try:
        from jax2onnx import to_onnx
    except ImportError as e:  # pragma: no cover - exercised only without jax2onnx
        raise ImportError(
            "ONNX export needs the optional 'onnx' extra: pip install 'sindae[onnx]'"
        ) from e

    fn = _ScaledMLP(net, bundle["scaler"]) if scaled else net
    to_onnx(
        fn=fn,
        inputs=[("B", bundle["in_size"])],
        enable_double_precision=True,
        return_mode="file",
        output_path=str(path),
    )
    sidecar = str(path) + ".json"
    with open(sidecar, "w") as f:
        json.dump(
            {
                "scaling": "internal" if scaled else "external",
                "scaler": _scaler_to_lists(bundle["scaler"]),
                "input_bounds": bundle["input_bounds"],
                "io_names": bundle["io_names"],
            },
            f, indent=2,
        )
    return path


# --------------------------------------------------------------------------- #
# Eval metrics functions. New metrics should be added to the _METRICS registry
# at the top of this file
# --------------------------------------------------------------------------- #

def _filter_data_from_collocation_points(problem: ProblemDefinition, data: InstanceData) -> tuple[list, list]:
    """Evaluate the predicted observed states at the observation times.

    The prediction lives on the trained collocation grid
    (``data.sampling_times``), which need not coincide with ``problem.obs_times``:
    training is routinely re-discretized to a different grid than the data was
    generated on.  For each trajectory the predicted observed states are
    therefore linearly interpolated onto the observation times, so the returned
    prediction lines up with ``problem.obs_values`` row-for-row regardless of
    grid.  When the observation times already fall on the collocation grid the
    interpolation is exact (returns the node values).
    """
    obs_times = problem.obs_times
    sampling_times = data.sampling_times
    pred_colloc = data.obs

    pred_at_obs = [
        np.column_stack([
            np.interp(obs_times[i], sampling_times[i], traj[:, k])
            for k in range(traj.shape[1])
        ])
        for i, traj in enumerate(pred_colloc)
    ]

    return problem.obs_values, pred_at_obs

def _compute_mse(x: list[np.ndarray], x_hat: list[np.ndarray]) -> np.ndarray:
    return np.array([list(np.mean((x_i - x_hat_i)**2, axis=0))
            for (x_i, x_hat_i) in zip(x, x_hat)])

def _compute_rmse(x: list[np.ndarray], x_hat: list[np.ndarray]) -> np.ndarray:
    return np.array([np.sqrt(np.mean((x_i - x_hat_i)**2, axis=0))
            for (x_i, x_hat_i) in zip(x, x_hat)])

def _compute_mae(x: list[np.ndarray], x_hat: list[np.ndarray]) -> np.ndarray:
    return np.array([list(np.mean(np.abs(x_i - x_hat_i), axis=0))
            for (x_i, x_hat_i) in zip(x, x_hat)])

_METRICS = {"mse": _compute_mse, 
            "rmse": _compute_rmse, 
            "mae": _compute_mae}