"""Tests for the high-level HybridDAE fit/predict wrapper (backlog item 3).

The wrapper's configuration surface is strict: the network comes in as a
prebuilt SimpleMLP (constructed outside the wrapper), stage configuration
comes in as config dataclasses (SmootherConfig / PretrainConfig /
SimultaneousConfig / DecompConfig / SolverConfig), never dicts, and everything
constructible is validated at construction so a typo fails before any solve.
The end-to-end tests run the full smoother -> pretrain -> train -> inference
pipeline on the small Leslie-Gower problem with the default POUNCE/FERAL
stack, mirroring the scale of tests/test_smoke.py.
"""
import jax
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

# The decomposition inner solve and the inference solve are grey-box
# (ExternalGreyBoxBlock) models: they build an in-memory PyomoNLP, which needs
# the compiled pynumero_ASL extension (`pyomo build-extensions`).  Individual
# end-to-end tests that touch those paths skip cleanly on a pip-only machine.
from pyomo.contrib.pynumero.asl import AmplInterface

needs_asl = pytest.mark.skipif(
    not AmplInterface.available(),
    reason="pynumero_ASL not available (run `pyomo build-extensions`)",
)


def _mlp(in_size=2, out_size=1, widths=(8, 8)):
    """Small SimpleMLP matching the Leslie-Gower problem by default."""
    from sindae import SimpleMLP

    return SimpleMLP(in_size, out_size, list(widths),
                     [jax.nn.softplus] * len(widths), key=jax.random.PRNGKey(0))


def _observed_problem(nfe=15, ncp=2):
    """Small Leslie-Gower problem with generated observations."""
    from sindae import generate_data
    from sindae.example_problems import LeslieGowerProblem

    problem = LeslieGowerProblem(nfe=nfe, ncp=ncp)
    data = generate_data(problem, noise_std=np.array([0.02, 0.02]),
                         obs_every=4, seed=0)
    assert data is not None, "data generation (true-model solve) failed"
    return problem


# ---------------------------------------------------------------------------
# Construction & validation (no solver)
# ---------------------------------------------------------------------------

def test_importable_from_package_with_defaults():
    from sindae import HybridDAE

    model = HybridDAE(net=_mlp())
    assert model.method == "simultaneous"
    assert model.nlp_solver == "pounce"
    assert model.linear_solver == "feral"


def test_net_must_be_a_prebuilt_simple_mlp():
    """The network is defined outside the wrapper; dicts and None are rejected."""
    from sindae import HybridDAE

    with pytest.raises(TypeError, match="SimpleMLP"):
        HybridDAE()  # net is required
    with pytest.raises(TypeError, match="SimpleMLP"):
        HybridDAE(net=dict(widths=[8, 8], activations="softplus"))


def test_invalid_method_raises():
    from sindae import HybridDAE

    with pytest.raises(ValueError, match="method"):
        HybridDAE(method="magic", net=_mlp())


def test_invalid_nlp_solver_raises():
    from sindae import HybridDAE

    with pytest.raises(ValueError, match="NLP solver"):
        HybridDAE(nlp_solver="ipoptt", net=_mlp())


def test_invalid_linear_solver_raises():
    from sindae import HybridDAE

    with pytest.raises(ValueError, match="linear"):
        HybridDAE(linear_solver="ma86", net=_mlp())


def test_stage_configs_must_be_config_objects_not_dicts():
    """The wrapper takes config dataclasses, never dicts, and says which."""
    from sindae import HybridDAE

    with pytest.raises(TypeError, match="SmootherConfig"):
        HybridDAE(net=_mlp(), smoother=dict(smooth_coef=2.0))
    with pytest.raises(TypeError, match="PretrainConfig"):
        HybridDAE(net=_mlp(), pretrain=dict(epochs=5))
    with pytest.raises(TypeError, match="SolverConfig"):
        HybridDAE(net=_mlp(), solver_options={"tol": 1e-6})
    with pytest.raises(TypeError, match="SimultaneousConfig"):
        HybridDAE(net=_mlp(), train=dict(reg_coef=1e-3))


def test_train_config_must_match_method():
    from sindae import DecompConfig, HybridDAE, SimultaneousConfig

    with pytest.raises(ValueError, match="method"):
        HybridDAE(method="simultaneous", net=_mlp(), train=DecompConfig())
    with pytest.raises(ValueError, match="method"):
        HybridDAE(method="decomposition", net=_mlp(), train=SimultaneousConfig())


def test_net_size_mismatch_raises_at_fit():
    from sindae import HybridDAE
    from sindae.example_problems import LeslieGowerProblem

    problem = LeslieGowerProblem(nfe=10, ncp=2)  # input_dim=2, z_dim=1
    model = HybridDAE(net=_mlp(in_size=3))
    with pytest.raises(ValueError, match="in_size"):
        model.fit(problem)


def test_predict_before_fit_raises():
    from sindae import HybridDAE
    from sindae.example_problems import LeslieGowerProblem

    model = HybridDAE(net=_mlp())
    with pytest.raises(RuntimeError, match="fit"):
        model.predict(LeslieGowerProblem(nfe=10, ncp=2))


def test_net_before_fit_raises():
    from sindae import HybridDAE

    with pytest.raises(RuntimeError, match="fit"):
        HybridDAE(net=_mlp()).net


def test_fit_requires_observation_data():
    from sindae import HybridDAE
    from sindae.example_problems import LeslieGowerProblem

    problem = LeslieGowerProblem(nfe=10, ncp=2)  # obs_times/obs_values unset
    with pytest.raises(ValueError, match="obs"):
        HybridDAE(net=_mlp()).fit(problem)


def test_pretrain_config_default_epochs_is_200():
    from sindae import PretrainConfig

    assert PretrainConfig().epochs == 200


# ---------------------------------------------------------------------------
# Stage-function plumbing needed by the wrapper (no solver)
# ---------------------------------------------------------------------------

def test_train_decomp_forwards_unfix_io(monkeypatch):
    """HybridDAE(unfix_io=...) must reach build_decomp_model, so train_decomp
    has to forward it (build_decomp_model already has the parameter)."""
    import sindae.algorithms.decomp.train as dtrain
    from sindae import DecompConfig, train_decomp
    from sindae.example_problems import LeslieGowerProblem

    class _Abort(Exception):
        pass

    captured = {}

    def fake_build(**kwargs):
        captured.update(kwargs)
        raise _Abort

    monkeypatch.setattr(dtrain, "build_decomp_model", fake_build)

    problem = LeslieGowerProblem(nfe=10, ncp=2)
    with pytest.raises(_Abort):
        train_decomp(problem, _mlp(widths=(4,)), DecompConfig(n_steps=1),
                     data=None, unfix_io=False)
    assert captured["unfix_io"] is False


def test_predict_does_not_inherit_training_solver_options(monkeypatch):
    """predict()'s inference solve defaults to solve_inference's own solver
    defaults (None), not the constructor's training solver_options, so a bare
    predict() matches a bare solve_inference() call.  An explicit override on
    predict still passes through."""
    import sindae.hybrid_dae as hd
    from sindae import HybridDAE, SolverConfig
    from sindae.example_problems import LeslieGowerProblem

    captured = {}

    class _Result:
        class solver:
            termination_condition = "optimal"

    class _Model:
        _solver_result = _Result()

    def fake_solve_inference(problem, net, data, **kwargs):
        captured.clear()
        captured.update(kwargs)
        return _Model()

    monkeypatch.setattr(hd, "solve_inference", fake_solve_inference)
    monkeypatch.setattr(hd, "extract_instance_data", lambda *a, **k: "data")

    mlp = _mlp()  # 2 -> 1, matches LeslieGower
    model = HybridDAE(net=mlp, solver_options=SolverConfig(tol=1e-9))
    model._net = mlp             # pretend fitted
    model.smoother_data = "sd"

    # No solver_options on predict -> inference gets None (POUNCE defaults),
    # NOT the constructor's training SolverConfig.
    model.predict(LeslieGowerProblem(nfe=10, ncp=2), slack_coef=1e-5)
    assert captured["solver_options"] is None

    # Explicit override still reaches the inference solve.
    opts = SolverConfig(tol=1e-3)
    model.predict(LeslieGowerProblem(nfe=10, ncp=2), slack_coef=1e-5,
                  solver_options=opts)
    assert captured["solver_options"] is opts


def test_fit_pretrains_by_default_and_warns_on_non_optimal(monkeypatch):
    """fit must run pretraining even when pretrain=None (with the 200-epoch
    default config), set model.termination, and warn on a non-optimal solve."""
    import sindae.hybrid_dae as hd
    from sindae import HybridDAE
    from sindae.example_problems import LeslieGowerProblem

    class _Result:
        class solver:
            termination_condition = "noSolution"

    class _Model:
        _solver_result = _Result()

    mlp = _mlp(widths=(4,))
    seen = {}

    def fake_pretrain(mlp, data, cfg):
        seen["cfg"] = cfg
        return mlp

    monkeypatch.setattr(hd, "solve_smoother", lambda *a, **k: _Model())
    monkeypatch.setattr(hd, "extract_instance_data", lambda *a, **k: "data")
    monkeypatch.setattr(hd, "pretrain_mlp", fake_pretrain)
    monkeypatch.setattr(hd, "solve_simultaneous", lambda *a, **k: (_Model(), mlp))

    problem = LeslieGowerProblem(nfe=10, ncp=2)
    problem.obs_times = [np.zeros(1)]
    problem.obs_values = [np.zeros((1, 2))]

    model = HybridDAE(net=mlp)
    with pytest.warns(UserWarning, match="noSolution"):
        model.fit(problem)
    assert model.termination == "noSolution"
    assert seen["cfg"].epochs == 200  # pretraining ran with the default config


# ---------------------------------------------------------------------------
# Persistence: save / load (net + scaler + architecture)
# ---------------------------------------------------------------------------

def _fake_fitted(mlp=None):
    """A HybridDAE with a simulated completed fit (no solver).

    Sets the trained network, a smoother_data carrying real normalization
    statistics, method and termination — the minimal state save() persists.
    """
    from sindae import HybridDAE, InstanceData, TrajectoryData

    if mlp is None:
        mlp = _mlp()
    traj = TrajectoryData(
        sampling_times=np.array([0.0, 1.0]),
        nn_input=np.array([[0.1, 0.2], [0.3, 0.5]]),
        nn_output=np.array([[1.0], [2.0]]),
        obs=np.array([[0.1, 0.2], [0.3, 0.5]]),
    )
    model = HybridDAE(net=mlp)
    model._net = mlp
    model.smoother_data = InstanceData([traj])
    model.termination = "optimal"
    return model


def test_save_requires_fitted(tmp_path):
    from sindae import HybridDAE

    model = HybridDAE(net=_mlp())
    with pytest.raises(RuntimeError, match="fit"):
        model.save(tmp_path / "m.eqx")


def test_load_is_classmethod_returning_fitted_wrapper(tmp_path):
    """HybridDAE.load(path) reconstructs a fitted wrapper without an instance."""
    from sindae import HybridDAE

    path = tmp_path / "m.eqx"
    _fake_fitted().save(path)

    loaded = HybridDAE.load(path)
    assert isinstance(loaded, HybridDAE)
    assert loaded.net is not None       # .net does not raise -> fitted
    loaded._check_fitted()              # guard passes


def test_save_load_roundtrip_preserves_weights_and_scaler(tmp_path):
    from sindae import HybridDAE
    from sindae.nn_utils import flatten_fn

    model = _fake_fitted()
    path = tmp_path / "m.eqx"
    model.save(path)
    loaded = HybridDAE.load(path)

    # Weights bit-for-bit.
    np.testing.assert_array_equal(flatten_fn(loaded.net), flatten_fn(model.net))
    # Architecture.
    assert loaded.net.in_size == model.net.in_size
    assert loaded.net.out_size == model.net.out_size
    assert list(loaded.net.widths) == list(model.net.widths)
    # Scaler (the four norm vectors solve_inference consumes).
    for attr in ("input_mean", "input_std", "output_mean", "output_std"):
        np.testing.assert_allclose(
            np.asarray(getattr(loaded.smoother_data, attr)),
            np.asarray(getattr(model.smoother_data, attr)),
        )
    # Metadata.
    assert loaded.method == model.method
    assert loaded.termination == "optimal"


def test_save_load_preserves_activation_strings(tmp_path):
    """Activations round-trip through their string names (mixed set)."""
    import jax.numpy as jnp

    from sindae import HybridDAE, SimpleMLP
    from sindae.nn_utils import _act_jax2str

    mlp = SimpleMLP(2, 1, [8, 8], [jnp.tanh, jax.nn.softplus],
                    key=jax.random.PRNGKey(1))
    path = tmp_path / "m.eqx"
    _fake_fitted(mlp).save(path)

    loaded = HybridDAE.load(path)
    assert [_act_jax2str(a) for a in loaded.net.activations] == ["tanh", "softplus"]


@needs_asl
def test_save_load_predict_matches_original(tmp_path):
    """A loaded model predicts bit-for-bit the same as the model that saved it."""
    from sindae import (
        HybridDAE,
        PretrainConfig,
        SimultaneousConfig,
        SmootherConfig,
        SolverConfig,
    )
    from sindae.example_problems import LeslieGowerProblem

    problem = _observed_problem()
    model = HybridDAE(
        net=_mlp(),
        smoother=SmootherConfig(smooth_coef=1.0),
        pretrain=PretrainConfig(epochs=5),
        train=SimultaneousConfig(reg_coef=1e-3),
        solver_options=SolverConfig(tol=1e-5, max_iter=200),
    )
    model.fit(problem)

    path = tmp_path / "m.eqx"
    model.save(path)
    loaded = HybridDAE.load(path)

    new_problem = LeslieGowerProblem(ics=np.array([[1.2, 0.15]]), nfe=15, ncp=2)
    ref = model.predict(new_problem, slack_coef=1e-5,
                        solver_options=SolverConfig(tol=1e-6))
    assert str(model.inference_model._solver_result.solver.termination_condition) \
        == "optimal"
    got = loaded.predict(new_problem, slack_coef=1e-5,
                        solver_options=SolverConfig(tol=1e-6))
    np.testing.assert_allclose(got[0].nn_output, ref[0].nn_output,
                               rtol=1e-6, atol=1e-8)


# ---------------------------------------------------------------------------
# End-to-end (POUNCE/FERAL default stack)
# ---------------------------------------------------------------------------

@needs_asl
def test_fit_predict_simultaneous_end_to_end():
    """fit hides smoother -> pretrain -> simultaneous; predict hides inference."""
    from sindae import (
        HybridDAE,
        InstanceData,
        PretrainConfig,
        SimpleMLP,
        SimultaneousConfig,
        SmootherConfig,
        SolverConfig,
    )
    from sindae.example_problems import LeslieGowerProblem

    problem = _observed_problem()

    model = HybridDAE(
        net=_mlp(),
        smoother=SmootherConfig(smooth_coef=1.0),
        pretrain=PretrainConfig(epochs=5),
        train=SimultaneousConfig(reg_coef=1e-3),
        solver_options=SolverConfig(tol=1e-5, max_iter=200),
    )
    fitted = model.fit(problem)
    assert fitted is model  # sklearn-style chaining

    assert isinstance(model.net, SimpleMLP)  # trained net, if you go looking
    assert isinstance(model.smoother_data, InstanceData)
    assert isinstance(model.trained_data, InstanceData)
    assert model.history is None  # simultaneous solve has no training loop
    assert model.termination == "optimal", (
        f"simultaneous solve did not reach optimal: {model.termination}"
    )

    # Predict on new initial conditions; no observations are needed.
    new_problem = LeslieGowerProblem(ics=np.array([[1.2, 0.15]]), nfe=15, ncp=2)
    pred = model.predict(new_problem, slack_coef=1e-5,
                         solver_options=SolverConfig(tol=1e-6))
    assert isinstance(pred, InstanceData)
    assert len(pred) == 1
    assert pred[0].nn_input.shape[1] == problem.input_dim
    assert np.all(np.isfinite(pred[0].nn_output))
    tc = str(model.inference_model._solver_result.solver.termination_condition)
    assert tc == "optimal", f"inference solve did not reach optimal: {tc}"


@needs_asl
def test_fit_decomposition_end_to_end():
    """fit runs smoother -> pretrain -> decomposition loop (POUNCE + FERAL)."""
    from sindae import DecompConfig, HybridDAE, PretrainConfig, SimpleMLP, SolverConfig

    problem = _observed_problem()

    model = HybridDAE(
        method="decomposition",
        net=_mlp(),
        pretrain=PretrainConfig(epochs=5),
        train=DecompConfig(n_steps=2, lr=1e-2, init_slack_coef=10.0),
        solver_options=SolverConfig(tol=1e-5, max_iter=200),
    )
    model.fit(problem)

    assert isinstance(model.net, SimpleMLP)
    assert model.termination is None  # per-step inner solves; see history
    assert len(model.history["obj_history"]) == 2
    assert np.all(np.isfinite(model.history["obj_history"]))
