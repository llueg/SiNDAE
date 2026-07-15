"""
timing_utils.py

Utilities for extracting IPOPT/POUNCE internal timing from solver output.

Both solvers print summary lines at the end of a run, e.g.

    Number of Iterations....: 12
    Total seconds in IPOPT (w/o function evaluations)   =  xx.xxx   (IPOPT)
    Total seconds in NLP function evaluations           =  xx.xxx   (IPOPT)
    Total seconds in POUNCE                             =  xx.xxx   (POUNCE)

These appear on the solver's standard output.  The POUNCE ASL executable
ignores IPOPT's ``output_file`` option, so the reliable capture is stdout
itself: the ASL solve path passes ``logfile=`` to ``solver.solve`` (pyomo
writes the subprocess stdout to that file regardless of ``tee``), while
cyipopt runs in-process and IPOPT honours ``output_file`` there (see
``set_output_file``).

API
---
  tmp_log_path() -> str
      Return a fresh temp-file path suitable for capturing a solver log.

  parse_pounce_output(lines) -> dict
      Parse ``pounceonly``, ``nlp_evals`` timing (seconds), ``n_iter`` and
      ``last_lgrg`` from solver output (a string or an iterable of lines).
      Returns None for any value that is not found (e.g. solve did not
      complete, or the backend does not print that line).

  parse_pounce_log(path) -> dict
      Same, reading the output from a log file (a missing/unreadable file
      yields all-None values).

  set_output_file(solver, path, is_cyipopt=False) -> None
      Set ``output_file`` / ``print_timing_statistics`` on a solver that
      honours them (IPOPT/cyipopt; the POUNCE executable ignores
      ``output_file``).
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
    """Return a new temp file path (not yet written to) for solver log output."""
    fd, path = tempfile.mkstemp(suffix='.log', prefix='pounce')
    os.close(fd)
    return path


def parse_pounce_output(lines) -> dict:
    """
    Parse IPOPT/POUNCE internal timing and iteration count from solver output.

    Parameters
    ----------
    lines : str or iterable of str
        The solver output — a multi-line string or an iterable of lines.

    Returns
    -------
    dict with keys:
      'pounceonly' : float or None  — seconds in the solver (IPOPT reports
                     this excluding NLP evals; POUNCE reports the total)
      'nlp_evals'  : float or None  — seconds in NLP function evaluations
                     (IPOPT only; POUNCE does not print this line)
      'n_iter'     : int or None    — iteration count, from the summary line
                     or, when that is absent, the last iteration-table row
      'last_lgrg'  : str or None    — lg(rg) at the last iteration ('-' when
                     no inertia regularization was applied)
    """
    if isinstance(lines, str):
        lines = lines.splitlines()

    pounceonly: Optional[float] = None
    nlp_evals:  Optional[float] = None
    n_iter:     Optional[int]   = None
    last_lgrg:  Optional[str]   = None   # lg(rg) value at the last iteration
    table_iter: Optional[int]   = None   # last iter number in the table

    for line in lines:
        m = re.search(
            r'Total (?:seconds|CPU secs) in (?:IPOPT|POUNCE)[^=]*=\s*([\d.]+)',
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

        m = _ITER_RE.match(line)
        if m:
            table_iter = int(m.group(1))
            last_lgrg = m.group(2)

    if n_iter is None:
        n_iter = table_iter

    return {
        'pounceonly': pounceonly,
        'nlp_evals':  nlp_evals,
        'n_iter':     n_iter,
        'last_lgrg':  last_lgrg,
    }


def parse_pounce_log(path: str) -> dict:
    """Parse solver timing/iteration info from a log file (see
    ``parse_pounce_output``).  A missing or unreadable file yields the
    all-None dict."""
    try:
        with open(path) as f:
            return parse_pounce_output(f)
    except OSError:
        return parse_pounce_output(())


def set_output_file(solver, path: str, is_cyipopt: bool = False) -> None:
    """Add ``output_file`` and ``print_timing_statistics`` to an IPOPT solver.

    ``print_timing_statistics=yes`` is required for IPOPT to emit the
    ``Total seconds in IPOPT`` / ``Total seconds in NLP function evaluations``
    lines that ``parse_pounce_log`` looks for.

    Only effective for solvers that honour ``output_file`` (IPOPT/cyipopt);
    the POUNCE ASL executable ignores it, so the ASL solve path captures the
    solver's stdout via pyomo's ``logfile=`` mechanism instead.
    """
    if is_cyipopt:
        solver.config.options['output_file'] = path
        solver.config.options['print_timing_statistics'] = 'yes'
    else:
        solver.options['output_file'] = path
        solver.options['print_timing_statistics'] = 'yes'
