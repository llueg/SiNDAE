"""
interior_point_compat.py

Provides InteriorPointInterface (as a subclass of Pyomo's version) that additionally
accepts a pre-built ExtendedNLP instance.

Standard Pyomo's InteriorPointInterface.__init__ only accepts a Pyomo ConcreteModel
or an .nl file path.  This subclass adds an ExtendedNLP branch so that
PyomoNLPWithGreyBoxBlocksExtended can be passed directly — replacing the small patch
that was previously applied to the local Pyomo clone's interface.py.
"""

import numpy as np
from pyomo.contrib.interior_point.interface import InteriorPointInterface as _Base
from pyomo.contrib.pynumero.interfaces.nlp import ExtendedNLP


class InteriorPointInterface(_Base):
    """
    InteriorPointInterface that additionally accepts ExtendedNLP instances.

    If ``model_or_nlp`` is an ``ExtendedNLP`` the ``_nlp`` attribute is set
    directly and the rest of the initialisation (slacks, bound duals, etc.) is
    performed inline.  Any other argument is forwarded to the standard Pyomo
    ``InteriorPointInterface.__init__``.
    """

    def __init__(self, model_or_nlp):
        if not isinstance(model_or_nlp, ExtendedNLP):
            super().__init__(model_or_nlp)
            return

        # ExtendedNLP path — mirrors the body of InteriorPointInterface.__init__
        # after _nlp is set (source: pyomo/contrib/interior_point/interface.py
        # as of Pyomo 6.10.0 / 6.10.1.dev0).
        self._nlp = model_or_nlp
        self._slacks = self.init_slacks()

        self._init_duals_primals_lb, self._init_duals_primals_ub = (
            self._get_full_duals_primals_bounds()
        )
        self._init_duals_primals_lb[np.isneginf(self._nlp.primals_lb())] = 0
        self._init_duals_primals_ub[np.isinf(self._nlp.primals_ub())] = 0
        self._duals_primals_lb = self._init_duals_primals_lb.copy()
        self._duals_primals_ub = self._init_duals_primals_ub.copy()

        self._init_duals_slacks_lb = self._nlp.init_duals_ineq().copy()
        self._init_duals_slacks_lb[self._init_duals_slacks_lb < 0] = 0
        self._init_duals_slacks_ub = self._nlp.init_duals_ineq().copy()
        self._init_duals_slacks_ub[self._init_duals_slacks_ub > 0] = 0
        self._init_duals_slacks_ub *= -1.0
        self._duals_slacks_lb = self._init_duals_slacks_lb.copy()
        self._duals_slacks_ub = self._init_duals_slacks_ub.copy()

        self._delta_primals = None
        self._delta_slacks = None
        self._delta_duals_eq = None
        self._delta_duals_ineq = None
        self._barrier = None
