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
    monkeypatch.setattr(hd, "_capture_io_names", lambda *a, **k: None)
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
    model.io_names = {"inputs": ["x[0]", "x[1]"], "outputs": ["z"]}
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
# Export: OMLT / ONNX / JSON (leaving SiNDAE for a foreign optimization tool)
# ---------------------------------------------------------------------------

import importlib.util


def _have(mod):
    return importlib.util.find_spec(mod) is not None


needs_omlt = pytest.mark.skipif(not _have("omlt"), reason="omlt not installed")
needs_onnx = pytest.mark.skipif(
    not (_have("jax2onnx") and _have("onnxruntime")),
    reason="jax2onnx / onnxruntime not installed",
)


def test_export_requires_fitted():
    from sindae import HybridDAE

    with pytest.raises(RuntimeError, match="fit"):
        HybridDAE(net=_mlp()).export(format="json")


def test_export_unknown_format_raises(tmp_path):
    model = _fake_fitted()
    with pytest.raises(ValueError, match="format"):
        model.export(tmp_path / "m.weird")
    with pytest.raises(ValueError, match="format"):
        model.export()  # no path, no format


def test_export_json_carries_weights_scaler_and_io_contract(tmp_path):
    """The JSON bundle is self-describing: weights, activations, the scaler, the
    data-derived input bounds, and the ordered IO variable-name contract."""
    import json

    from sindae.nn_utils import flatten_fn

    model = _fake_fitted()
    path = tmp_path / "m.json"
    ret = model.export(path)
    assert ret == path

    bundle = json.loads(path.read_text())

    # Architecture + activation names (hidden layers only; output is linear).
    assert bundle["in_size"] == 2 and bundle["out_size"] == 1
    assert bundle["activations"] == ["softplus", "softplus"]

    # Weights round-trip: rebuilding the flat vector matches the trained net.
    flat = np.concatenate(
        [np.asarray(w).ravel() for w in bundle["weights"]]
        + [np.asarray(b).ravel() for b in bundle["biases"]]
    )
    # (order-independent check: same multiset of parameter values)
    np.testing.assert_allclose(np.sort(flat),
                               np.sort(np.asarray(flatten_fn(model.net))))

    # Scaler is the four normalization vectors.
    for attr in ("input_mean", "input_std", "output_mean", "output_std"):
        np.testing.assert_allclose(
            np.asarray(bundle["scaler"][attr]),
            np.asarray(getattr(model.smoother_data, attr)),
        )

    # IO contract (positional variable names) and data-derived input bounds.
    assert bundle["io_names"] == {"inputs": ["x[0]", "x[1]"], "outputs": ["z"]}
    assert len(bundle["input_bounds"]) == 2


@needs_onnx
def test_export_onnx_matches_jax(tmp_path):
    """The exported ONNX graph reproduces the JAX network under onnxruntime
    (external oracle), and a scaler sidecar is written alongside it."""
    import json

    import jax.numpy as jnp
    import onnxruntime as ort

    model = _fake_fitted()
    path = tmp_path / "m.onnx"
    model.export(path)

    assert path.exists()
    sidecar = tmp_path / "m.onnx.json"
    assert sidecar.exists()
    meta = json.loads(sidecar.read_text())
    assert "scaler" in meta and "io_names" in meta
    # Default export leaves the graph in normalized space; the scaler is applied
    # externally by the consumer, so the sidecar advertises that contract.
    assert meta["scaling"] == "external"

    sess = ort.InferenceSession(str(path))
    iname = sess.get_inputs()[0].name
    x = np.array([[1.3, -0.4], [0.2, 0.9], [-1.0, 2.0]])
    onnx_out = sess.run(None, {iname: x})[0]
    ref = np.asarray(jax.vmap(model.net)(jnp.array(x)))
    np.testing.assert_allclose(onnx_out, ref, rtol=1e-6, atol=1e-8)


@needs_onnx
def test_export_onnx_scaled_bakes_scaler(tmp_path):
    """With scaled=True the ONNX graph consumes RAW physical inputs and returns
    RAW physical outputs: it reproduces normalize -> net -> denormalize end to
    end (external oracle), and the sidecar marks the scaling as internal."""
    import json

    import jax.numpy as jnp
    import onnxruntime as ort

    model = _fake_fitted()
    path = tmp_path / "m_scaled.onnx"
    ret = model.export(path, scaled=True)
    assert str(ret) == str(path)

    sidecar = tmp_path / "m_scaled.onnx.json"
    assert sidecar.exists()
    meta = json.loads(sidecar.read_text())
    assert meta["scaling"] == "internal"

    sess = ort.InferenceSession(str(path))
    iname = sess.get_inputs()[0].name
    x = np.array([[1.3, -0.4], [0.2, 0.9], [-1.0, 2.0]])
    onnx_out = sess.run(None, {iname: x})[0]

    sd = model.smoother_data
    in_mean, in_std = jnp.asarray(sd.input_mean), jnp.asarray(sd.input_std)
    out_mean, out_std = jnp.asarray(sd.output_mean), jnp.asarray(sd.output_std)

    def physical(v):
        return out_mean + out_std * model.net((v - in_mean) / in_std)

    ref = np.asarray(jax.vmap(physical)(jnp.array(x)))
    np.testing.assert_allclose(onnx_out, ref, rtol=1e-6, atol=1e-8)


@needs_onnx
def test_export_onnx_scaled_only_valid_for_onnx(tmp_path):
    """scaled=True is an ONNX-only knob; it is rejected for json export."""
    model = _fake_fitted()
    with pytest.raises(ValueError, match="scaled"):
        model.export(tmp_path / "m.json", scaled=True)


@needs_omlt
def test_export_omlt_reproduces_normalized_forward():
    """OMLT NetworkDefinition + OffsetScaling maps raw physical inputs to raw
    physical outputs, matching the full normalize -> net -> denormalize path.

    External oracle: a Pyomo solve of the OMLT formulation with fixed inputs is
    compared to the SimpleMLP evaluated through the training normalization.
    """
    import jax.numpy as jnp
    import pyomo.environ as pyo
    from omlt import OmltBlock
    from omlt.neuralnet import FullSpaceSmoothNNFormulation

    model = _fake_fitted()
    net_def = model.to_omlt()

    m = pyo.ConcreteModel()
    m.nn = OmltBlock()
    m.nn.build_formulation(FullSpaceSmoothNNFormulation(net_def))
    x_raw = np.array([0.25, 0.35])
    m.nn.inputs[0].fix(x_raw[0])
    m.nn.inputs[1].fix(x_raw[1])
    m.obj = pyo.Objective(expr=0.0)
    pyo.SolverFactory("pounce").solve(m)
    omlt_out = pyo.value(m.nn.outputs[0])

    sd = model.smoother_data
    xn = (x_raw - np.asarray(sd.input_mean)) / np.asarray(sd.input_std)
    z = np.asarray(jax.vmap(model.net)(jnp.array(xn[None])))[0]
    ref = np.asarray(sd.output_mean) + np.asarray(sd.output_std) * z
    np.testing.assert_allclose(omlt_out, float(ref[0]), rtol=1e-6, atol=1e-8)


@needs_asl
def test_export_omlt_after_fit_and_load_captures_io_names(tmp_path):
    """A model fit on LeslieGower captures the IO variable-name contract, and it
    survives a save/load round-trip into the export bundle."""
    import json

    from sindae import HybridDAE, PretrainConfig, SimultaneousConfig, SolverConfig

    problem = _observed_problem()
    model = HybridDAE(
        net=_mlp(),
        pretrain=PretrainConfig(epochs=5),
        train=SimultaneousConfig(reg_coef=1e-3),
        solver_options=SolverConfig(tol=1e-5, max_iter=200),
    )
    model.fit(problem)
    # LeslieGower inputs are x[0], x[1]; output is z.
    assert model.io_names == {"inputs": ["x[0]", "x[1]"], "outputs": ["z"]}

    save_path = tmp_path / "m.eqx"
    model.save(save_path)
    reloaded = HybridDAE.load(save_path)
    assert reloaded.io_names == model.io_names

    json_path = tmp_path / "m.json"
    reloaded.export(json_path)
    assert json.loads(json_path.read_text())["io_names"] == model.io_names


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
