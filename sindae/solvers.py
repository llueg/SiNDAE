"""
solvers.py

Backend selection for the two solver roles in SiNDAE.

NLP solver (the optimiser that solves each Pyomo model)
    POUNCE is the pip-installable default; IPOPT and cyipopt are selectable
    alternatives.  ``make_nlp_solver(backend)`` returns an :class:`NLPSolver`
    that hides the differences between the ASL-based solvers (POUNCE, IPOPT;
    options set on ``solver.options``) and cyipopt (options set on
    ``solver.config.options``).  POUNCE handles grey-box
    (``ExternalGreyBoxBlock``) models too — including the ``return_nlp=True``
    solve the decomposition inner loop relies on — via its cyipopt-style Python
    interface (see :mod:`sindae.interfaces.pounce_interface`); cyipopt remains a
    selectable alternative for those models, not the default.

Linear / KKT solver (the sparse symmetric solver inside the decomposition
gradient back-solve)
    FERAL is the pip-installable default; MA27 and scipy are selectable
    alternatives.  They already share Pyomo's ``IPLinearSolverInterface``
    protocol, so ``make_linear_solver(name)`` is just a constructor selector.

The stage functions (``solve_smoother``, ``solve_simultaneous``,
``solve_inference``, ``train_decomp``) accept these selectors so a backend can
be chosen without monkeypatching; the high-level ``HybridDAE`` wrapper layers a
string facade on top of them.
"""
from __future__ import annotations

import abc
import logging
import os
from dataclasses import dataclass
from typing import Optional, Union

import pyomo.environ as pyo

from sindae.algorithms.timing_utils import (
    parse_pounce_log,
    set_output_file,
    tmp_log_path,
)

logger = logging.getLogger(__name__)


@dataclass
class NLPResult:
    """Outcome of one :meth:`NLPSolver.solve` call.

    Attributes
    ----------
    result : pyomo SolverResults
        The object returned by ``solver.solve`` (carries
        ``solver.termination_condition`` etc.).
    timing : dict
        Parsed solver timing/iteration info — IPOPT or POUNCE branded output
        (see ``parse_pounce_output``); values are None when the backend does
        not print the corresponding line.
    nlp : object or None
        The populated NLP returned by ``return_nlp=True`` (cyipopt, or
        POUNCE on grey-box models); ``None`` otherwise.
    """

    result: object
    timing: dict
    nlp: object = None


class NLPSolver(abc.ABC):
    """Abstract NLP backend wrapping a configured Pyomo solver.

    Concrete subclasses set ``name`` (the ``SolverFactory`` name) and, when
    applicable, the ``is_cyipopt`` / ``supports_return_nlp`` capability flags.
    The underlying Pyomo solver is built once at construction and reused across
    :meth:`solve` calls (matching the decomposition loop, which solves the same
    model hundreds of times).
    """

    #: cyipopt sets options on ``solver.config.options``; ASL solvers on
    #: ``solver.options``.  Also selects the log-capture code path.
    is_cyipopt: bool = False
    #: Whether ``solve(return_nlp=True)`` can return the populated NLP
    #: (cyipopt always; POUNCE for grey-box models only).
    supports_return_nlp: bool = False

    @property
    @abc.abstractmethod
    def name(self) -> str:
        """The ``SolverFactory`` name for this backend."""

    def __init__(self, options: Optional[dict] = None):
        self._options = dict(options or {})
        self._solver = pyo.SolverFactory(self.name)
        self._apply_options(self._options)

    @property
    def pyomo_solver(self):
        """The underlying configured Pyomo solver (for advanced tuning)."""
        return self._solver

    def _apply_options(self, options: dict) -> None:
        target = self._solver.config.options if self.is_cyipopt else self._solver.options
        for key, value in options.items():
            target[key] = value

    def solve(
        self,
        model,
        *,
        tee: bool = False,
        extra_options: Optional[dict] = None,
        return_nlp: bool = False,
    ) -> NLPResult:
        """Solve ``model``, capturing solver timing from its log output.

        The ASL executables do not all honour IPOPT's ``output_file`` option
        (POUNCE ignores it), so on the ASL path the subprocess stdout is
        captured to a temporary file via pyomo's ``logfile=`` mechanism
        (``tee=True`` still streams to the console).  cyipopt runs in-process
        and IPOPT itself writes the log via ``output_file``.

        Parameters
        ----------
        model : pyomo model
        tee : bool
            Stream solver output to stdout.
        extra_options : dict, optional
            Per-call options overlaid on the persistent solver options for
            this solve only (the persistent options are left untouched).
        return_nlp : bool
            Return the populated NLP alongside the results (raises
            ``ValueError`` for backends that do not support it).
        """
        if return_nlp and not self.supports_return_nlp:
            raise ValueError(
                f"NLP backend {self.name!r} does not support return_nlp=True"
            )

        log_path = tmp_log_path()
        # The ``options`` kwarg is ephemeral in pyomo — overlaid on the
        # persistent options for this call only (OptSolver restores them;
        # cyipopt builds a per-call config) — so extra_options cannot leak
        # into later solves.
        solve_kwargs = {"tee": tee}
        if self.is_cyipopt:
            set_output_file(self._solver, log_path, is_cyipopt=True)
            if extra_options:
                solve_kwargs["options"] = dict(extra_options)
        else:
            solve_kwargs["logfile"] = log_path
            solve_kwargs["options"] = {
                "print_timing_statistics": "yes",
                **(extra_options or {}),
            }
        try:
            if return_nlp:
                result, nlp = self._solver.solve(
                    model, return_nlp=True, **solve_kwargs
                )
            else:
                result = self._solver.solve(model, **solve_kwargs)
                nlp = None
            timing = parse_pounce_log(log_path)
        finally:
            try:
                os.unlink(log_path)
            except OSError:
                pass

        return NLPResult(result=result, timing=timing, nlp=nlp)


class PounceSolver(NLPSolver):
    """POUNCE: pip-installable default NLP backend.

    Non-grey-box models go through the ASL ``SolverFactory('pounce')`` path
    (NL/SOL files).  Models carrying an ``ExternalGreyBoxBlock`` are routed
    through POUNCE's cyipopt-style Python interface
    (:func:`sindae.interfaces.pounce_interface.solve_pyomo_with_pounce`), which
    is the only path that can consume grey-box callbacks and is what enables the
    ``return_nlp=True`` solve used by the decomposition inner loop.
    """

    name = "pounce"
    supports_return_nlp = True

    @staticmethod
    def _has_grey_box(model) -> bool:
        from pyomo.contrib.pynumero.interfaces.external_grey_box import (
            ExternalGreyBoxBlock,
        )

        return any(model.component_data_objects(ExternalGreyBoxBlock, active=True))

    def solve(
        self,
        model,
        *,
        tee: bool = False,
        extra_options: Optional[dict] = None,
        return_nlp: bool = False,
    ) -> NLPResult:
        if self._has_grey_box(model):
            from sindae.interfaces.pounce_interface import solve_pyomo_with_pounce

            options = {**self._options, **(extra_options or {})}
            result, nlp, timing = solve_pyomo_with_pounce(
                model, options=options, tee=tee, return_nlp=return_nlp
            )
            return NLPResult(result=result, timing=timing, nlp=nlp)

        if return_nlp:
            raise ValueError(
                "Pounce can only return the NLP for grey-box models "
                "(ExternalGreyBoxBlock); this model has none"
            )
        return super().solve(model, tee=tee, extra_options=extra_options)


class IpoptSolver(NLPSolver):
    """IPOPT (ASL): conda/HSL alternative NLP backend."""

    name = "ipopt"


class CyIpoptSolver(NLPSolver):
    """cyipopt: required for grey-box (ExternalGreyBoxBlock) models and the
    decomposition inner solve (``return_nlp=True``)."""

    name = "cyipopt"
    is_cyipopt = True
    supports_return_nlp = True


_NLP_BACKENDS = {
    "pounce": PounceSolver,
    "ipopt": IpoptSolver,
    "cyipopt": CyIpoptSolver,
}


def make_nlp_solver(
    backend: Union[str, NLPSolver] = "pounce",
    options: Optional[dict] = None,
) -> NLPSolver:
    """Build an :class:`NLPSolver` for ``backend``.

    ``backend`` may be a name (``"pounce"`` (default), ``"ipopt"``,
    ``"cyipopt"``; case-insensitive) or an existing :class:`NLPSolver`, which is
    returned unchanged (``options`` are ignored in that case — a warning is
    logged if any were passed).
    """
    if isinstance(backend, NLPSolver):
        if options:
            logger.warning(
                "make_nlp_solver: backend is an already-constructed %s "
                "instance; ignoring options %s",
                type(backend).__name__,
                sorted(options),
            )
        return backend
    try:
        cls = _NLP_BACKENDS[backend.lower()]
    except (AttributeError, KeyError):
        raise ValueError(
            f"Unknown NLP backend {backend!r}; choose from "
            f"{sorted(_NLP_BACKENDS)}"
        )
    return cls(options=options)


def make_linear_solver(name: Union[str, object] = "feral", **kwargs):
    """Build a linear / KKT solver implementing Pyomo's
    ``IPLinearSolverInterface``.

    ``name`` may be ``"feral"`` (default), ``"ma27"``, ``"scipy"``
    (case-insensitive) or an already-constructed interface, which is returned
    unchanged (keyword arguments are ignored in that case — a warning is
    logged if any were passed).  Extra keyword arguments are forwarded to the
    constructor (used by FERAL's ``max_steps`` / ``refine_tol`` /
    ``residual_tol``).
    """
    if not isinstance(name, str):
        if kwargs:
            logger.warning(
                "make_linear_solver: name is an already-constructed %s "
                "instance; ignoring keyword arguments %s",
                type(name).__name__,
                sorted(kwargs),
            )
        return name

    key = name.lower()
    if key == "feral":
        from sindae.interfaces.feral_interface import FeralInterface

        return FeralInterface(**kwargs)
    if key == "ma27":
        from pyomo.contrib.interior_point.linalg.ma27_interface import (
            InteriorPointMA27Interface,
        )

        return InteriorPointMA27Interface(**kwargs)
    if key == "scipy":
        from pyomo.contrib.interior_point.linalg.scipy_interface import ScipyInterface

        return ScipyInterface(**kwargs)

    raise ValueError(
        f"Unknown linear solver {name!r}; choose from ['feral', 'ma27', 'scipy']"
    )
