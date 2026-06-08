"""
Pytest configuration for the SiNDAE solver-parity tests.

The pounce ASL CLI ships *inside* the ``pounce-solver`` wheel and is installed
next to the Python interpreter (e.g. ``<env>/bin/pounce``).  When the conda env
is not "activated" in the calling shell, that directory may not be on ``PATH``,
so Pyomo's ``SolverFactory('pounce')`` can't find the executable.  Prepend it
here so the tests work regardless of how the interpreter was launched.
"""
import os
import sys

_bindir = os.path.dirname(sys.executable)
if _bindir and _bindir not in os.environ.get("PATH", "").split(os.pathsep):
    os.environ["PATH"] = _bindir + os.pathsep + os.environ.get("PATH", "")
