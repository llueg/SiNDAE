from enum import Enum
import copy
import logging
from unittest.mock import patch

import jax
jax.config.update('jax_enable_x64', True)

import optax
import pytest
import numpy as np
import pyomo.environ as pyo
from dataclasses import dataclass

from sindae.nn_utils import SimpleMLP
from sindae import generate_data
from sindae.data_utils import extract_instance_data, InstanceData
from sindae.algorithms.smoother import solve_smoother
from sindae.algorithms.pretrain import PretrainConfig, pretrain_mlp
from sindae.algorithms.decomp.train import DecompConfig, train_decomp
from sindae.algorithms.simultaneous.train import solve_simultaneous
from sindae.algorithms.simultaneous.model_builder import extract_mlp

from sindae.example_problems import FourTankProblem, FedBatchBioreactorProblem, LeslieGowerProblem

logger = logging.getLogger(__name__)

X_TOL = 1e-3
Z_TOL = 1e-3

@dataclass
class TestConfig:
    # Problem init
    problem: FourTankProblem | LeslieGowerProblem | FedBatchBioreactorProblem
    # MLP init
    mlp: SimpleMLP
    # Data gen init
    noise_std: np.ndarray
    seed: int
    # Smooth solve init
    nfe_train: int
    ncp_train: int
    smooth_coef: float
    # Pretrain init
    reg_coef: float
    epochs: int
    batch_size: int


class Configs(Enum):
    four_tank = TestConfig(
        # Problem init
        problem=FourTankProblem(
            nfe=40, 
            ncp=3
        ),
        # MLP init
        mlp=SimpleMLP(
            in_size=4,  
            out_size=2, 
            widths=[32, 32], 
            activations=[jax.nn.tanh] * 2,
            key=jax.random.PRNGKey(0),
        ),
        # Data gen init
        noise_std=np.array([0.05, 0.05, 0.05, 0.05]),
        seed = 0,
        # Smooth solve init
        nfe_train = 20,
        ncp_train = 2,
        smooth_coef = 1e1,
        # Pretrain init
        reg_coef = 1e-2,
        epochs = 200,
        batch_size = 32,
    )

    leslie_gower = TestConfig(
        # Problem init
        problem=LeslieGowerProblem(
            nfe=60, 
            ncp=3
        ),
        # MLP init
        mlp=SimpleMLP(
            in_size=2,  
            out_size=1, 
            widths=[16, 16], 
            activations=[jax.nn.softplus] * 2,
            key=jax.random.PRNGKey(0),
        ),
        # Data gen init
        noise_std=np.array([0.05, 0.05]),
        seed = 0,
        # Smooth solve init
        nfe_train = 40,
        ncp_train = 3,
        smooth_coef = 1e1, 
        # Pretrain init
        reg_coef = 1e-3,
        epochs = 200,
        batch_size = 32,
    )

    fedbatch = TestConfig(
        # Problem init
        problem=FedBatchBioreactorProblem(
            nfe=40, 
            ncp=3
        ),
        # MLP init
        mlp=SimpleMLP(
            in_size=4,  
            out_size=1, 
            widths=[20, 20], 
            activations=[jax.nn.softplus] * 2,
            key=jax.random.PRNGKey(0),
        ),
        # Data gen init
        seed = 0,
        noise_std=np.array([0.05, 0.05, 0.5, 0.1]),
        # Smooth solve init
        nfe_train = 20,
        ncp_train = 3,
        smooth_coef = 1e1,
        # Pretrain init
        reg_coef = 1e-3,
        epochs = 200,
        batch_size = 32,
    )
    

def solve_simultaneous_with(
    config: TestConfig, 
    solver_name: str, 
    use_gbm: bool, 
    hess_approx: str
) -> tuple:
    """
    Solve single config with specified solver and options, using monkeypatching 
    to swap solvers without modifying the main codebase.
    
    Returns
    -------
    trained_m : pyo.ConcreteModel or None
        The solved simultaneous model (None if skipped).
    trained_mlp : SimpleMLP or None
        The trained MLP with optimized weights (None if skipped).
    instance_data : InstanceData or None
        Extracted data from the trained model (None if skipped).
    """
    logger.info(f"Solving with {solver_name} (use_gbm={use_gbm}, hess_approx={hess_approx})")
    
    # Skip incompatible combinations
    if use_gbm and hess_approx != 'limited-memory':
        logger.warning(f"  Skipping {solver_name} with use_gbm=True and hess_approx='exact'")
        return None, None, None
    
    # Recreate MLP with fresh random key
    problem = copy.deepcopy(config.problem)
    mlp = SimpleMLP(
        in_size=config.mlp.in_size,
        out_size=config.mlp.out_size,
        widths=config.mlp.widths,
        activations=config.mlp.activations,
        key=jax.random.PRNGKey(config.seed),
    )
    
    # ── 1. Generate data ──────────────────────────────────────────────────────
    logger.info(f'  1. Generating data for {solver_name}...')
    true_data: InstanceData = generate_data(
        problem=problem, 
        noise_std=config.noise_std, 
        obs_every=4, 
        seed=config.seed,
    )
    
    # ── 2. Solve smoother ─────────────────────────────────────────────────────
    logger.info(f'  2. Solving smoother for {solver_name}...')
    problem.nfe = config.nfe_train
    problem.ncp = config.ncp_train
    smoother_m = solve_smoother(
        problem, 
        mlp, 
        smooth_coef=config.smooth_coef
    )
    smoother_data: InstanceData = extract_instance_data(problem, smoother_m)
    
    # ── 3. Pretrain MLP ───────────────────────────────────────────────────────
    logger.info(f'  3. Pretraining MLP for {solver_name}...')
    mlp = pretrain_mlp(
        mlp, 
        smoother_data, 
        PretrainConfig(
            epochs=config.epochs, 
            batch_size=config.batch_size, 
            reg_coef=config.reg_coef
        )
    )
    
    # ── 4. Solve simultaneously with solver swapping ────────────────────────────
    logger.info(f'  4. Solving simultaneously with {solver_name}...')
    
    ipopt_options = {'hessian_approximation': hess_approx}
    
    # Determine which default solver path (ipopt or cyipopt) and patch accordingly
    if use_gbm:
        default_solver = 'cyipopt'
    else:
        default_solver = 'ipopt'
    
    # Monkeypatch SolverFactory to redirect the call to our target solver
    original_solver_factory = pyo.SolverFactory
    
    def patched_solver_factory(solver_name_arg, **kwargs):
        """Intercept solver factory calls and use our target solver."""
        if solver_name_arg in (default_solver, 'ipopt', 'cyipopt'):
            # Redirect the default solver request to our target solver
            return original_solver_factory(solver_name, **kwargs)
        return original_solver_factory(solver_name_arg, **kwargs)
    
    try:
        # Patch in both pyomo.environ and the train module
        with patch('pyomo.environ.SolverFactory', side_effect=patched_solver_factory):
            with patch('sindae.algorithms.simultaneous.train.pyo.SolverFactory', side_effect=patched_solver_factory):
                trained_m, trained_mlp = solve_simultaneous(
                    problem=problem, 
                    mlp=mlp,
                    data=smoother_data, 
                    smoother_model=smoother_m,
                    use_gbm=use_gbm, 
                    reg_coef=config.reg_coef,
                    ipopt_options=ipopt_options,
                    tee=False,
                )
    except Exception as e:
        logger.error(f"  Error solving with {solver_name}: {e}")
        raise
    
    # ── 5. Extract trained data ────────────────────────────────────────────────
    instance_data: InstanceData = extract_instance_data(problem, trained_m)
    
    logger.info(f'  ✓ Completed {solver_name}')
    return trained_m, trained_mlp, instance_data


# Test case to solve all three problems with ipopt/pounce and check results match.
def test_pounce_matches_ipopt_simultaneous():
    """
    Test that POUNCE produces results within tolerance of IPOPT for the 
    simultaneous method (expr-writing, no GBM) across all test configurations.
    
    Note: This test only covers use_gbm=False since POUNCE is an ASL-based solver
    like IPOPT, not a cyipopt replacement. For GBM models, cyipopt is required.
    """
    for config_enum in Configs:
        logger.info(f"\n{'='*80}")
        logger.info(f"Testing config: {config_enum.name}")
        logger.info(f"{'='*80}")
        
        test_config = config_enum.value
        
        # Only test with use_gbm=False (expr-writing)
        # For GBM (use_gbm=True), cyipopt is required and pounce cannot be used
        use_gbm = False
        
        for hess_approx in ['exact', 'limited-memory']:
            logger.info(f"\n  Testing use_gbm={use_gbm}, hess_approx='{hess_approx}'")
            
            # Solve with reference solver (ipopt)
            ref_m, ref_mlp, ref_data = solve_simultaneous_with(
                test_config, 
                'ipopt', 
                use_gbm, 
                hess_approx
            )
            
            if ref_data is None:
                logger.info(f"    Skipped ipopt due to incompatible settings")
                continue
            
            # Solve with pounce
            pounce_m, pounce_mlp, pounce_data = solve_simultaneous_with(
                test_config, 
                'pounce', 
                use_gbm, 
                hess_approx
            )
            
            if pounce_data is None:
                logger.info(f"    Skipped pounce due to incompatible settings")
                continue
            
            # ── Compare results ───────────────────────────────────────────
            logger.info(f"    Comparing results between ipopt and pounce...")
            
            # Extract NN outputs for all trajectories
            ref_z = np.concatenate([traj.nn_output for traj in ref_data._trajectories])
            pounce_z = np.concatenate([traj.nn_output for traj in pounce_data._trajectories])
            
            # Extract observed states for all trajectories  
            ref_x = np.concatenate([traj.obs for traj in ref_data._trajectories])
            pounce_x = np.concatenate([traj.obs for traj in pounce_data._trajectories])
            
            # Calculate relative RMSE
            z_rmse = relative_rmse(ref_z, pounce_z)
            x_rmse = relative_rmse(ref_x, pounce_x)
            
            logger.info(f"      NN output relative RMSE: {z_rmse:.6e}")
            logger.info(f"      State obs relative RMSE: {x_rmse:.6e}")
            logger.info(f"      Tolerance (Z_TOL): {Z_TOL:.6e}, (X_TOL): {X_TOL:.6e}")
            
            # Assert within tolerances
            assert z_rmse < Z_TOL, (
                f"NN output mismatch for {config_enum.name} "
                f"(use_gbm={use_gbm}, hess_approx='{hess_approx}'): "
                f"RMSE={z_rmse:.6e} > tolerance={Z_TOL:.6e}"
            )
            
            assert x_rmse < X_TOL, (
                f"State obs mismatch for {config_enum.name} "
                f"(use_gbm={use_gbm}, hess_approx='{hess_approx}'): "
                f"RMSE={x_rmse:.6e} > tolerance={X_TOL:.6e}"
            )
            
            logger.info(f"    ✓ PASSED: {config_enum.name} with use_gbm={use_gbm}, hess_approx='{hess_approx}'")
    
        logger.info(f"\n✓ All tests passed for {config_enum.name}")


def relative_rmse(ref, other):
    return np.sqrt(np.mean((ref - other) ** 2)) / (np.sqrt(np.mean(ref ** 2)) + 1e-8)
    