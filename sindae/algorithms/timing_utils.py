"""
timing_utils.py

Utilities for extracting IPOPT internal timing from solver log files.

IPOPT always prints two summary lines at the end of its run:

    Total seconds in IPOPT (w/o function evaluations)   =  xx.xxx
    Total seconds in NLP function evaluations           =  xx.xxx

These appear in both the standard output stream and in any ``output_file``
set via solver options.  Using ``output_file`` is the most reliable way to
capture them programmatically regardless of the ``tee`` setting.

API
---
  tmp_log_path() -> str
      Return a fresh temp-file path suitable for ``output_file``.

  parse_pounce_log(path) -> dict
      Parse ``pounceonly``, ``nlp_evals`` timing (seconds) and ``n_iter``
      from a log.  Returns None for any value that is not found (e.g. solve
      did not complete, or a very old IPOPT version).

  capture_pouncetiming(solver, is_cyipopt=False) -> (path, cleanup)
      Context-manager-free helper: sets output_file on the solver and
      returns the log path.  Caller must call cleanup() after parsing.
"""
from __future__ import annotations

import os
import re
import tempfile
from typing import Optional

# Matches the iteration index at the start of an IPOPT iteration row.  The
# trailing letter marks restoration-phase / acceptance iterations (e.g. "12r").
# Fields: iter  obj  inf_pr  inf_du  lg(mu)  ||d||  lg(rg)  alpha_du alpha_pr  ls
_ITER_HEAD_RE = re.compile(r'^(\d+)[a-zA-Z]?$')


def _parse_iter_row(line: str):
    """Parse one IPOPT iteration row.

    Returns ``(iter, objective, inf_du, lg_rg)`` or ``None`` if the line is not
    an iteration row.  ``objective`` is the current objective value and
    ``inf_du`` is the scaled dual infeasibility (the Lagrangian-gradient norm).
    """
    parts = line.split()
    if len(parts) < 7 or not _ITER_HEAD_RE.match(parts[0]):
        return None
    try:
        objective = float(parts[1])
        inf_du    = float(parts[3])
    except ValueError:
        return None
    return int(_ITER_HEAD_RE.match(parts[0]).group(1)), objective, inf_du, parts[6]


def tmp_log_path() -> str:
    """Return a new temp file path (not yet written to) for IPOPT log output."""
    fd, path = tempfile.mkstemp(suffix='.log', prefix='pounce')
    os.close(fd)
    return path


def parse_pounce_log(path: str) -> dict:
    """
    Parse IPOPT internal timing and iteration count from a log file.

    Returns
    -------
    dict with keys:
      'pounceonly'        : float or None  — seconds in IPOPT excluding NLP evals
      'nlp_evals'         : float or None  — seconds in NLP function evaluations
      'n_iter'            : int or None    — number of IPOPT iterations
      'last_lgrg'         : str or None    — lg(rg) at the final iteration
      'obj_history'       : list[float]    — objective value per iteration
      'grad_norm_history' : list[float]    — scaled dual infeasibility per iteration
    """
    pounceonly: Optional[float] = None
    nlp_evals:  Optional[float] = None
    n_iter:     Optional[int]   = None
    last_lgrg:  Optional[str]   = None
    obj_history:       list = []
    grad_norm_history: list = []

    try:
        with open(path) as f:
            for line in f:
                m = re.search(
                    r'Total (?:seconds|CPU secs) in IPOPT[^=]*=\s*([\d.]+)',
                    line,
                )
                if m:
                    pounceonly = float(m.group(1))

                m = re.search(
                    r'Total (?:seconds|CPU secs) in NLP function evaluations\s*=\s*([\d.]+)',
                    line,
                )
                if m:
                    nlp_evals = float(m.group(1))

                m = re.search(r'Number of Iterations\.+:\s*(\d+)', line)
                if m:
                    n_iter = int(m.group(1))

                row = _parse_iter_row(line)
                if row is not None:
                    _, objective, inf_du, last_lgrg = row
                    obj_history.append(objective)
                    grad_norm_history.append(inf_du)
    except OSError:
        pass

    return {
        'pounceonly':        pounceonly,
        'nlp_evals':         nlp_evals,
        'n_iter':            n_iter,
        'last_lgrg':         last_lgrg,
        'obj_history':       obj_history,
        'grad_norm_history': grad_norm_history,
    }


def set_output_file(solver, path: str, is_cyipopt: bool = False) -> None:
    """Add ``output_file`` and ``print_timing_statistics`` to an IPOPT solver.

    ``print_timing_statistics=yes`` is required for IPOPT to emit the
    ``Total seconds in IPOPT`` / ``Total seconds in NLP function evaluations``
    lines that ``parse_pounce_log`` looks for.
    """
    if is_cyipopt:
        solver.config.options['output_file'] = path
        solver.config.options['print_timing_statistics'] = 'yes'
    else:
        solver.options['output_file'] = path
        solver.options['print_timing_statistics'] = 'yes'
