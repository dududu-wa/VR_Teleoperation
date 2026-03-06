"""
Evaluation visualization helpers.
"""

from __future__ import annotations

from typing import Dict, Sequence, Optional
import matplotlib.pyplot as plt
import numpy as np


def _prepare_axis(ax, title: str, xlabel: str, ylabel: str):
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.25)


def plot_eval_curves(
    x_values: Sequence[float],
    metrics: Dict[str, Sequence[float]],
    title: str = "Evaluation Curves",
    save_path: Optional[str] = None,
):
    """Plot one or multiple metric curves against a common x-axis."""
    x = np.asarray(x_values, dtype=float)
    fig, ax = plt.subplots(figsize=(7, 4))
    for name, y_values in metrics.items():
        y = np.asarray(y_values, dtype=float)
        ax.plot(x, y, marker="o", label=name)
    _prepare_axis(ax, title, "Setting", "Metric")
    ax.legend()
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150)
    return fig, ax


def plot_phase_metrics(
    phase_ids: Sequence[int],
    fall_rate: Sequence[float],
    tracking_score: Sequence[float],
    transition_failure: Sequence[float],
    save_path: Optional[str] = None,
):
    """Plot key curriculum metrics per phase."""
    phase = np.asarray(phase_ids, dtype=int)
    fr = np.asarray(fall_rate, dtype=float)
    tr = np.asarray(tracking_score, dtype=float)
    tf = np.asarray(transition_failure, dtype=float)

    fig, axes = plt.subplots(1, 3, figsize=(12, 3.5), sharex=True)
    axes[0].plot(phase, fr, marker="o", color="#d62728")
    _prepare_axis(axes[0], "Fall Rate", "Phase", "Rate")

    axes[1].plot(phase, tr, marker="o", color="#2ca02c")
    _prepare_axis(axes[1], "Tracking Score", "Phase", "Score")

    axes[2].plot(phase, tf, marker="o", color="#1f77b4")
    _prepare_axis(axes[2], "Transition Failure", "Phase", "Rate")

    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150)
    return fig, axes

