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

  parse_ipopt_log(path) -> dict
      Parse ``ipopt_only``, ``nlp_evals`` timing (seconds) and ``n_iter``
      from a log.  Returns None for any value that is not found (e.g. solve
      did not complete, or a very old IPOPT version).

  capture_ipopt_timing(solver, is_cyipopt=False) -> (path, cleanup)
      Context-manager-free helper: sets output_file on the solver and
      returns the log path.  Caller must call cleanup() after parsing.
"""
from __future__ import annotations

import os
import re
import tempfile
from typing import Optional

# Matches one IPOPT iteration row.  The 7th field (index 6) is lg(rg).
# Fields: iter  obj  inf_pr  inf_du  lg(mu)  ||d||  lg(rg)  ...
_ITER_RE = re.compile(
    r'^\s*(\d+)\s+'   # iter number
    r'\S+\s+'         # objective
    r'\S+\s+'         # inf_pr
    r'\S+\s+'         # inf_du
    r'\S+\s+'         # lg(mu)
    r'\S+\s+'         # ||d||
    r'(\S+)'          # lg(rg): '-' or a decimal like '-4.0'
)


def tmp_log_path() -> str:
    """Return a new temp file path (not yet written to) for IPOPT log output."""
    fd, path = tempfile.mkstemp(suffix='.log', prefix='ipopt_')
    os.close(fd)
    return path


def parse_ipopt_log(path: str) -> dict:
    """
    Parse IPOPT internal timing and iteration count from a log file.

    Returns
    -------
    dict with keys:
      'ipopt_only' : float or None  — seconds in IPOPT excluding NLP evals
      'nlp_evals'  : float or None  — seconds in NLP function evaluations
      'n_iter'     : int or None    — number of IPOPT iterations
    """
    ipopt_only: Optional[float] = None
    nlp_evals:  Optional[float] = None
    n_iter:     Optional[int]   = None
    last_lgrg:  Optional[str]   = None   # lg(rg) value at the last iteration

    try:
        with open(path) as f:
            for line in f:
                m = re.search(
                    r'Total (?:seconds|CPU secs) in IPOPT[^=]*=\s*([\d.]+)',
                    line,
                )
                if m:
                    ipopt_only = float(m.group(1))

                m = re.search(
                    r'Total (?:seconds|CPU secs) in NLP function evaluations\s*=\s*([\d.]+)',
                    line,
                )
                if m:
                    nlp_evals = float(m.group(1))

                m = re.search(r'Number of Iterations\.+:\s*(\d+)', line)
                if m:
                    n_iter = int(m.group(1))

                m = _ITER_RE.match(line)
                if m:
                    last_lgrg = m.group(2)
    except OSError:
        pass

    return {
        'ipopt_only': ipopt_only,
        'nlp_evals':  nlp_evals,
        'n_iter':     n_iter,
        'last_lgrg':  last_lgrg,
    }


def set_output_file(solver, path: str, is_cyipopt: bool = False) -> None:
    """Add ``output_file`` and ``print_timing_statistics`` to an IPOPT solver.

    ``print_timing_statistics=yes`` is required for IPOPT to emit the
    ``Total seconds in IPOPT`` / ``Total seconds in NLP function evaluations``
    lines that ``parse_ipopt_log`` looks for.
    """
    if is_cyipopt:
        solver.config.options['output_file'] = path
        solver.config.options['print_timing_statistics'] = 'yes'
    else:
        solver.options['output_file'] = path
        solver.options['print_timing_statistics'] = 'yes'
