# ____________________________________________________________________________________
#
# Pyomo: Python Optimization Modeling Objects
# Copyright (c) 2008-2026 National Technology and Engineering Solutions of Sandia, LLC
# Under the terms of Contract DE-NA0003525 with National Technology and Engineering
# Solutions of Sandia, LLC, the U.S. Government retains certain rights in this
# software.  This software is distributed under the 3-clause BSD License.
# ____________________________________________________________________________________
"""
Extends PyomoNLPWithGreyBoxBlocks with the ExtendedNLP interface, which
separates equality and inequality constraints.

Typical usage after a cyipopt solve::

    results, nlp = solver.solve(m, return_nlp=True)
    nlp_ext = PyomoNLPWithGreyBoxBlocksExtended(nlp)
    iface = InteriorPointInterface(nlp_ext)
"""

import numpy as np
from scipy.sparse import coo_matrix

from pyomo.contrib.pynumero.interfaces.nlp import ExtendedNLP
from pyomo.contrib.pynumero.interfaces.pyomo_grey_box_nlp import (
    PyomoNLPWithGreyBoxBlocks,
)


class PyomoNLPWithGreyBoxBlocksExtended(ExtendedNLP):
    """Wraps a PyomoNLPWithGreyBoxBlocks and adds the ExtendedNLP interface.

    All grey box constraints are equalities (lb == ub == 0).  The Pyomo
    part may contain both equality and inequality constraints.  This class
    builds the eq/ineq split by inspecting constraints_lb() / constraints_ub()
    on the wrapped NLP (same logic as AslNLP._build_constraint_maps).

    Parameters
    ----------
    nlp : PyomoNLPWithGreyBoxBlocks
        A fully constructed (and typically solved) grey-box NLP instance.
        Obtain one via ``solver.solve(model, return_nlp=True)``.
    """

    def __init__(self, nlp):
        super().__init__()
        if not isinstance(nlp, PyomoNLPWithGreyBoxBlocks):
            raise TypeError(
                'PyomoNLPWithGreyBoxBlocksExtended requires a '
                'PyomoNLPWithGreyBoxBlocks instance, got '
                f'{type(nlp).__name__}'
            )
        self._nlp = nlp
        self._build_constraint_maps()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_constraint_maps(self):
        """Build eq/ineq masks and cached jacobian structures."""
        con_lb = self._nlp.constraints_lb()
        con_ub = self._nlp.constraints_ub()
        bounds_diff = np.abs(con_ub - con_lb)
        tol = 1e-8

        self._con_full_eq_mask = bounds_diff < tol
        self._con_eq_full_map = self._con_full_eq_mask.nonzero()[0]
        self._con_full_ineq_mask = ~self._con_full_eq_mask
        self._con_ineq_full_map = self._con_full_ineq_mask.nonzero()[0]
        self._n_con_eq = len(self._con_eq_full_map)
        self._n_con_ineq = len(self._con_ineq_full_map)

        # Build full-index → eq-index and full-index → ineq-index maps
        n_con = self._nlp.n_constraints()
        self._full_to_eq = np.full(n_con, -1, dtype=np.intp)
        self._full_to_eq[self._con_eq_full_map] = np.arange(
            self._n_con_eq, dtype=np.intp
        )
        self._full_to_ineq = np.full(n_con, -1, dtype=np.intp)
        self._full_to_ineq[self._con_ineq_full_map] = np.arange(
            self._n_con_ineq, dtype=np.intp
        )

        # Evaluate the full jacobian once to determine sparsity structure.
        # The structure (row, col indices) is assumed fixed across evaluations.
        jac_full = self._nlp.evaluate_jacobian()
        self._nz_eq_mask = np.isin(jac_full.row, self._con_eq_full_map)
        self._nz_ineq_mask = ~self._nz_eq_mask

        n_primals = self._nlp.n_primals()
        eq_rows = self._full_to_eq[jac_full.row[self._nz_eq_mask]]
        eq_cols = jac_full.col[self._nz_eq_mask]
        ineq_rows = self._full_to_ineq[jac_full.row[self._nz_ineq_mask]]
        ineq_cols = jac_full.col[self._nz_ineq_mask]

        self._cached_jac_eq = coo_matrix(
            (jac_full.data[self._nz_eq_mask].copy(), (eq_rows, eq_cols)),
            shape=(self._n_con_eq, n_primals),
        )
        self._cached_jac_ineq = coo_matrix(
            (jac_full.data[self._nz_ineq_mask].copy(), (ineq_rows, ineq_cols)),
            shape=(self._n_con_ineq, n_primals),
        )
        self._nnz_jac_eq = int(self._nz_eq_mask.sum())
        self._nnz_jac_ineq = int(self._nz_ineq_mask.sum())

    # ------------------------------------------------------------------
    # NLP abstract methods — delegated to self._nlp
    # ------------------------------------------------------------------

    def n_primals(self):
        return self._nlp.n_primals()

    def n_constraints(self):
        return self._nlp.n_constraints()

    def nnz_jacobian(self):
        return self._nlp.nnz_jacobian()

    def nnz_hessian_lag(self):
        return self._nlp.nnz_hessian_lag()

    def primals_lb(self):
        return self._nlp.primals_lb()

    def primals_ub(self):
        return self._nlp.primals_ub()

    def constraints_lb(self):
        return self._nlp.constraints_lb()

    def constraints_ub(self):
        return self._nlp.constraints_ub()

    def init_primals(self):
        return self._nlp.init_primals()

    def init_duals(self):
        return self._nlp.init_duals()

    def create_new_vector(self, vector_type):
        if vector_type == 'primals':
            return np.zeros(self.n_primals(), dtype=np.float64)
        elif vector_type in ('constraints', 'duals'):
            return np.zeros(self.n_constraints(), dtype=np.float64)
        elif vector_type in ('eq_constraints', 'duals_eq'):
            return np.zeros(self.n_eq_constraints(), dtype=np.float64)
        elif vector_type in ('ineq_constraints', 'duals_ineq'):
            return np.zeros(self.n_ineq_constraints(), dtype=np.float64)
        else:
            raise RuntimeError(
                f'Called create_new_vector with an unknown vector_type: {vector_type}'
            )

    def set_primals(self, primals):
        self._nlp.set_primals(primals)

    def get_primals(self):
        return self._nlp.get_primals()

    def set_duals(self, duals):
        self._nlp.set_duals(duals)

    def get_duals(self):
        return self._nlp.get_duals()

    def set_obj_factor(self, obj_factor):
        self._nlp.set_obj_factor(obj_factor)

    def get_obj_factor(self):
        return self._nlp.get_obj_factor()

    def get_obj_scaling(self):
        return self._nlp.get_obj_scaling()

    def get_primals_scaling(self):
        return self._nlp.get_primals_scaling()

    def get_constraints_scaling(self):
        return self._nlp.get_constraints_scaling()

    def evaluate_objective(self):
        return self._nlp.evaluate_objective()

    def evaluate_grad_objective(self, out=None):
        return self._nlp.evaluate_grad_objective(out=out)

    def evaluate_constraints(self, out=None):
        return self._nlp.evaluate_constraints(out=out)

    def evaluate_jacobian(self, out=None):
        return self._nlp.evaluate_jacobian(out=out)

    def evaluate_hessian_lag(self, out=None):
        return self._nlp.evaluate_hessian_lag(out=out)

    def report_solver_status(self, status_code, status_message):
        return self._nlp.report_solver_status(status_code, status_message)

    def primals_names(self):
        return self._nlp.primals_names()

    def constraint_names(self):
        return self._nlp.constraint_names()

    def get_primal_indices(self, pyomo_variables):
        """Return NLP primal indices for a list of Pyomo VarData objects."""
        name_to_idx = {n: i for i, n in enumerate(self._nlp.primals_names())}
        return [name_to_idx[v.name] for v in pyomo_variables]

    def get_constraint_indices(self, pyomo_constraints):
        """Return full NLP constraint indices for a list of Pyomo ConstraintData objects.

        Note: grey-box output constraints have no Pyomo ConstraintData object and
        therefore cannot be queried through this method.
        """
        name_to_idx = {n: i for i, n in enumerate(self._nlp.constraint_names())}
        return [name_to_idx[c.name] for c in pyomo_constraints]

    def get_grey_box_output_constraint_indices(self, block):
        """Return eq-block indices for the output constraints of an ExternalGreyBoxBlock.

        Grey-box output constraints are named
        ``'{block_fqn}.output_constraints[{output_name}]'``
        and are always equalities, so the returned indices are directly usable with
        ``get_duals_eq()`` and the KKT rho vector eq-block offset.

        Parameters
        ----------
        block : ExternalGreyBoxBlock
            The grey-box block whose output constraint indices are requested.

        Returns
        -------
        list[int]
            Eq-block indices in the same order as
            ``block.get_external_model().output_names()``.
        """
        ex_model = block.get_external_model()
        prefix = block.getname(fully_qualified=True)
        name_to_full_idx = {n: i for i, n in enumerate(self._nlp.constraint_names())}
        full_indices = np.array(
            [
                name_to_full_idx['{}.output_constraints[{}]'.format(prefix, nm)]
                for nm in ex_model.output_names()
            ],
            dtype=np.intp,
        )
        return list(self._full_to_eq[full_indices])

    # ------------------------------------------------------------------
    # ExtendedNLP abstract methods
    # ------------------------------------------------------------------

    def n_eq_constraints(self):
        return self._n_con_eq

    def n_ineq_constraints(self):
        return self._n_con_ineq

    def nnz_jacobian_eq(self):
        return self._nnz_jac_eq

    def nnz_jacobian_ineq(self):
        return self._nnz_jac_ineq

    def ineq_lb(self):
        return self._nlp.constraints_lb()[self._con_full_ineq_mask]

    def ineq_ub(self):
        return self._nlp.constraints_ub()[self._con_full_ineq_mask]

    def init_duals_eq(self):
        return self._nlp.init_duals()[self._con_full_eq_mask]

    def init_duals_ineq(self):
        return self._nlp.init_duals()[self._con_full_ineq_mask]

    def set_duals_eq(self, duals_eq):
        duals = self._nlp.get_duals()
        duals[self._con_full_eq_mask] = duals_eq
        self._nlp.set_duals(duals)

    def get_duals_eq(self):
        return self._nlp.get_duals()[self._con_full_eq_mask]

    def set_duals_ineq(self, duals_ineq):
        duals = self._nlp.get_duals()
        duals[self._con_full_ineq_mask] = duals_ineq
        self._nlp.set_duals(duals)

    def get_duals_ineq(self):
        return self._nlp.get_duals()[self._con_full_ineq_mask]

    def get_eq_constraints_scaling(self):
        scaling = self._nlp.get_constraints_scaling()
        if scaling is not None:
            return scaling[self._con_full_eq_mask]
        return None

    def get_ineq_constraints_scaling(self):
        scaling = self._nlp.get_constraints_scaling()
        if scaling is not None:
            return scaling[self._con_full_ineq_mask]
        return None

    def evaluate_eq_constraints(self, out=None):
        eq_c = self._nlp.evaluate_constraints()[self._con_full_eq_mask]
        if out is not None:
            np.copyto(out, eq_c)
            return out
        return eq_c

    def evaluate_ineq_constraints(self, out=None):
        ineq_c = self._nlp.evaluate_constraints()[self._con_full_ineq_mask]
        if out is not None:
            np.copyto(out, ineq_c)
            return out
        return ineq_c

    def evaluate_jacobian_eq(self, out=None):
        jac_full = self._nlp.evaluate_jacobian()
        np.copyto(self._cached_jac_eq.data, jac_full.data[self._nz_eq_mask])
        if out is not None:
            np.copyto(out.data, self._cached_jac_eq.data)
            return out
        return self._cached_jac_eq.copy()

    def evaluate_jacobian_ineq(self, out=None):
        jac_full = self._nlp.evaluate_jacobian()
        np.copyto(self._cached_jac_ineq.data, jac_full.data[self._nz_ineq_mask])
        if out is not None:
            np.copyto(out.data, self._cached_jac_ineq.data)
            return out
        return self._cached_jac_ineq.copy()

    # ------------------------------------------------------------------
    # Convenience pass-throughs (used by InteriorPointInterface)
    # ------------------------------------------------------------------

    def load_state_into_pyomo(self, bound_multipliers=None):
        self._nlp.load_state_into_pyomo(bound_multipliers=bound_multipliers)

    def pyomo_model(self):
        """Return the underlying Pyomo model."""
        return self._nlp._pyomo_model

    def get_pyomo_variables(self):
        """Return the ordered list of Pyomo VarData objects."""
        return self._nlp._pyomo_model_var_datas
