"""Fast smoke tests for the CI per-push gate.

These run in a few seconds and catch the common breakages: a broken import, a
packaging mistake, or a regression that stops the core pipeline from solving.
"""
import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)


def test_public_api_imports():
    from sindae import (  # noqa: F401
        SimpleMLP,
        ProblemDefinition,
        InstanceData,
        generate_data,
        extract_instance_data,
        SimultaneousConfig,
        solve_simultaneous,
        DecompConfig,
        train_decomp,
    )


def test_simplemlp_forward_and_roundtrip():
    from sindae import SimpleMLP
    from sindae.nn_utils import flatten_fn, make_unflatten_fn

    mlp = SimpleMLP(
        in_size=3, out_size=2, widths=[8, 8],
        activations=[jax.nn.softplus] * 2, key=jax.random.PRNGKey(0),
    )
    x = jnp.ones(3)
    assert mlp(x).shape == (2,)

    flat = flatten_fn(mlp)
    mlp2 = make_unflatten_fn(mlp)(flat)
    assert jnp.allclose(mlp(x), mlp2(x))


def test_example_problems_construct():
    from sindae.example_problems import LeslieGowerProblem, FedBatchBioreactorProblem

    lg = LeslieGowerProblem(nfe=10, ncp=2)
    assert (lg.input_dim, lg.z_dim) == (2, 1)

    fb = FedBatchBioreactorProblem(nfe=10, ncp=2)
    assert (fb.input_dim, fb.z_dim) == (4, 1)


def test_end_to_end_simultaneous():
    """data -> smoother -> simultaneous solve (POUNCE). No cyipopt needed."""
    from sindae import SimpleMLP, generate_data, extract_instance_data, SimultaneousConfig
    from sindae.algorithms.smoother import solve_smoother
    from sindae.algorithms.simultaneous.train import solve_simultaneous
    from sindae.example_problems import LeslieGowerProblem

    problem = LeslieGowerProblem(nfe=15, ncp=2)
    mlp = SimpleMLP(2, 1, [8, 8], [jax.nn.softplus] * 2, key=jax.random.PRNGKey(0))

    data = generate_data(problem, noise_std=np.array([0.02, 0.02]), obs_every=4, seed=0)
    assert data is not None, "data generation (true-model solve) failed"

    smoother_m = solve_smoother(problem, mlp, smooth_coef=1.0)
    smoother_data = extract_instance_data(problem, smoother_m)

    cfg = SimultaneousConfig(use_gbm=False, reg_coef=1e-3)
    trained_m, mlp = solve_simultaneous(
        problem, mlp, cfg, data=smoother_data, smoother_model=smoother_m,
        solver_options={"tol": 1e-5, "max_iter": 200},
    )

    tc = str(trained_m._solver_result.solver.termination_condition)
    assert tc == "optimal", f"simultaneous solve did not reach optimal: {tc}"

    trained = extract_instance_data(problem, trained_m)
    assert np.all(np.isfinite(trained[0].nn_output))
