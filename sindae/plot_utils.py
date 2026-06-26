"""
plot_utils.py  —  trajectory plotting utilities for InstanceData

Functions
---------
plot_instance_data(datasets, nn_input_names, nn_output_names, ...) -> (fig, axes)
    Plot nn_input / nn_output from one or more InstanceData objects on a
    (num_traj × num_vars) grid.  Optionally overlays observed data as scatter.

plot_training_history(history) -> (fig, axes)
    Plot objective and gradient-norm history from train_decomp.

Usage example
-------------
    from plot_utils import plot_instance_data, plot_training_history

    fig, axes = plot_instance_data(
        datasets=[
            (true_data,     'true',     {'color': 'black', 'ls': '-'}),
            (smoother_data, 'smoother', {'color': 'C0',    'ls': '--'}),
            (decomp_data,   'decomp',   {'color': 'C1',    'ls': '-.'}),
        ],
        nn_input_names=['x0', 'x1'],
        nn_output_names=['z'],
        obs_times=problem.obs_times,
        obs_values=problem.obs_values,
        obs_names=['x0', 'x1'],   # obs col 0 → 'x0' panel, obs col 1 → 'x1' panel
    )
    fig.savefig('trajectories.png')
"""
from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import matplotlib.pyplot as plt

from sindae.data_utils import InstanceData

plt.rcParams['figure.dpi'] = 300
plt.rcParams['font.size'] = 14
plt.rcParams['font.family'] = 'sans-serif'
# Arial preferred (paper figures); fall back to DejaVu Sans (matplotlib's bundled
# default) on systems without Arial — e.g. Binder — to avoid findfont warnings.
plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans']
plt.rcParams['axes.labelsize'] = 14
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['xtick.labelsize'] = 12
plt.rcParams['ytick.labelsize'] = 12
plt.rcParams['legend.fontsize'] = 14
plt.rcParams['lines.linewidth'] = 2
plt.rcParams['lines.markersize'] = 5
marker_linewidth = 1

#set colormap
plt.rcParams['axes.prop_cycle'] = plt.cycler(color=plt.cm.Dark2.colors)


# Default line style cycle (overridden per dataset via the line_kwargs argument)
_DEFAULT_STYLES: List[Dict] = [
    {
        'color': 'C0',
        'ls': '-',
        #'lw': 1.5
        },
    {
        'color': 'C1',
        'ls': '--', 
        #'lw': 1.2
        },
    {
        'color': 'C2',
        'ls': '-.',
        #'lw': 1.2
        },
    {
        'color': 'C3',
        'ls': ':',
        #'lw': 1.2
        },
    {
        'color': 'C4',
        'ls': '-',
        #'lw': 1.0
        },
]


def plot_instance_data(
    datasets: Sequence[Tuple[InstanceData, str, Optional[Dict]]],
    nn_input_names: List[str],
    nn_output_names: List[str],
    aux_var_names: Optional[List[str]] = None,
    groups: Optional[List[str]] = None,
    obs_times: Optional[List[np.ndarray]] = None,
    obs_values: Optional[List[np.ndarray]] = None,
    obs_names: Optional[List[str]] = None,
    traj_labels: Optional[List[str]] = None,
    fig: Optional[plt.Figure] = None,
    axes: Optional[np.ndarray] = None,
    figsize_per_panel: Tuple[float, float] = (4.0, 3.0),
    legend_placement: Optional[str] = 'first',
    legend_kwargs: Optional[Dict] = None,
    var_names_as_ylabel: bool = False,
) -> Tuple[plt.Figure, np.ndarray]:
    """
    Plot trajectory data from one or more InstanceData objects on a grid.

    Grid layout: rows = trajectories, columns selected by ``groups``.

    Parameters
    ----------
    datasets : list of (InstanceData, label, line_kwargs)
        Each entry is plotted as a line series.  ``line_kwargs`` (dict or None)
        overrides the default style for that dataset.
    nn_input_names : List[str]
        Names for ``traj.nn_input`` columns.
    nn_output_names : List[str]
        Names for ``traj.nn_output`` columns.
    aux_var_names : List[str], optional
        Names for ``traj.aux_vars`` columns.  Required to show the 'aux' group.
    groups : List[str], optional
        Which variable groups to include, e.g. ``['inputs', 'outputs']``,
        ``['aux']``, or ``['inputs', 'outputs', 'aux']``.
        Defaults to all groups that have names provided.
    obs_times : List[np.ndarray], optional
        Per-trajectory observation times for scatter overlay.
    obs_values : List[np.ndarray], optional
        Per-trajectory observation arrays, each shape ``(n_obs, obs_dim)``.
    obs_names : List[str], optional
        Name for each observation column.  Matches against column names in the
        figure; unrecognised names are silently ignored.
    traj_labels : List[str], optional
        Row labels placed as y-axis labels on the first column.
    fig, axes : optional
        Existing figure / axes to plot into (shape ``(num_traj, num_cols)``).
    figsize_per_panel : (float, float)
        ``(width, height)`` per subplot panel in inches.
    legend_placement : str, optional
        Where to place the legend.  One of:
        - 'first': place in first panel (default)
        - 'last': place in last panel
    legend_kwargs : dict, optional
        Forwarded to ``ax.legend()``.

    Returns
    -------
    fig  : plt.Figure
    axes : np.ndarray, shape (num_traj, num_cols)
    """
    if not datasets:
        raise ValueError("datasets must be non-empty")

    # ── Determine which groups to show ────────────────────────────────────────
    _avail = ['inputs', 'outputs']
    if aux_var_names is not None:
        _avail.append('aux')
    if groups is None:
        groups = _avail

    # ── Build ordered column list: (display_name, group, within_group_idx) ────
    _cols: List[Tuple[str, str, int]] = []
    if 'inputs' in groups:
        _cols += [(name, 'inputs', j) for j, name in enumerate(nn_input_names)]
    if 'outputs' in groups:
        _cols += [(name, 'outputs', k) for k, name in enumerate(nn_output_names)]
    if 'aux' in groups and aux_var_names is not None:
        _cols += [(name, 'aux', m) for m, name in enumerate(aux_var_names)]

    if not _cols:
        raise ValueError("No columns to plot — check groups / name lists")

    all_names = [c[0] for c in _cols]
    num_cols  = len(all_names)
    num_traj  = datasets[0][0].num_trajectories

    # name → column index (for obs scatter routing)
    name_to_col: Dict[str, int] = {name: c for c, (name, _, _) in enumerate(_cols)}

    if traj_labels is None:
        traj_labels = [f'Trajectory {i + 1}' for i in range(num_traj)]

    # ── Create figure ─────────────────────────────────────────────────────────
    if fig is None or axes is None:
        w, h = figsize_per_panel
        fig, axes = plt.subplots(
            num_traj, num_cols,
            figsize=(w * num_cols, h * num_traj),
            squeeze=False,
        )
    axes = np.asarray(axes)
    if axes.ndim == 1:
        axes = axes.reshape(num_traj, num_cols)

    # ── Line plots ─────────────────────────────────────────────────────────────
    for ds_idx, (data, label, kw) in enumerate(datasets):
        style = dict(_DEFAULT_STYLES[ds_idx % len(_DEFAULT_STYLES)])
        if kw:
            style.update(kw)

        for i in range(num_traj):
            traj = data[i]
            t    = traj.sampling_times

            for c_idx, (_, group, grp_idx) in enumerate(_cols):
                if group == 'inputs':
                    y = traj.nn_input[:, grp_idx]
                elif group == 'outputs':
                    y = traj.nn_output[:, grp_idx]
                else:  # 'aux'
                    if traj.aux_vars is None:
                        continue
                    y = traj.aux_vars[:, grp_idx]
                axes[i, c_idx].plot(t, y, label=label, **style)

    if legend_placement == 'first':
        legend_traj, legend_col = 0, 0
    elif legend_placement == 'last':
        legend_traj = num_traj - 1
        legend_col = num_cols - 1
    else:
        legend_traj, legend_col = None, None

    # ── Scatter: observed data ─────────────────────────────────────────────────
    if obs_times is not None and obs_values is not None and obs_names is not None:
        for i in range(num_traj):
            t_obs = obs_times[i]
            v_obs = np.asarray(obs_values[i])
            if v_obs.ndim == 1:
                v_obs = v_obs[:, None]

            for k, oname in enumerate(obs_names):
                if oname not in name_to_col:
                    continue
                col = name_to_col[oname]
                obs_label = 'data' if (i == legend_traj and col == legend_col) else None
                axes[i, col].scatter(
                    t_obs, v_obs[:, k],
                    s=18, zorder=5, color='red', marker='x',
                    linewidths=1.0, label=obs_label,
                )

    # ── Axis decorations ───────────────────────────────────────────────────────
    if var_names_as_ylabel:
        for i in range(num_traj):
            for c, name in enumerate(all_names):
                axes[i, c].set_ylabel(name)
    else:
        for c, name in enumerate(all_names):
            axes[0, c].set_title(name)
        for i, row_label in enumerate(traj_labels):
            axes[i, 0].set_ylabel(row_label)

    for c in range(num_cols):
        axes[num_traj - 1, c].set_xlabel('$t$')

    # ── Legend: deduplicate handles on the chosen panel ────────────────────────
    if legend_placement is not None:
        lkw = {'loc': 'best'}
        if legend_kwargs:
            lkw.update(legend_kwargs)
        leg_ax = axes[legend_traj, legend_col]
        handles, labels = leg_ax.get_legend_handles_labels()
        seen: Dict[str, int] = {}
        for idx, lbl in enumerate(labels):
            if lbl not in seen:
                seen[lbl] = idx

            leg_ax.legend(
                [handles[i] for i in seen.values()],
                list(seen.keys()),
                **lkw,
            )

    fig.tight_layout()
    return fig, axes


def plot_training_history(
    history: dict,
    fig: Optional[plt.Figure] = None,
    axes: Optional[np.ndarray] = None,
) -> Tuple[plt.Figure, np.ndarray]:
    """
    Plot objective and gradient-norm history from ``train_decomp``.

    Parameters
    ----------
    history : dict
        Keys ``'obj_history'`` and ``'grad_norm_history'`` (as returned by
        ``train_decomp``).
    fig, axes : optional
        Existing figure / axes array of length ≥ 2.  If ``None``, a new 1×2
        figure is created.

    Returns
    -------
    fig  : plt.Figure
    axes : np.ndarray, shape (2,)
    """
    obj_hist  = history['obj_history']
    grad_hist = history['grad_norm_history']
    steps     = np.arange(1, len(obj_hist) + 1)

    if fig is None or axes is None:
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    axes = np.asarray(axes).ravel()
    axes[0].semilogy(steps, obj_hist)
    axes[0].set_xlabel('step')
    axes[0].set_title('Objective')
    axes[1].semilogy(steps, grad_hist)
    axes[1].set_xlabel('step')
    axes[1].set_title('Gradient norm')

    fig.tight_layout()
    return fig, np.asarray(axes)
