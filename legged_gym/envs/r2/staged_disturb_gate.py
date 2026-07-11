"""Simulator-independent gate policy for staged disturbance curricula."""

from typing import Mapping, Optional


WindowStats = Mapping[str, float]
ProfileStats = Mapping[str, WindowStats]


def validate_profile_gate_resampling(
    *,
    require_all_profiles: bool,
    resampling_time_s: float,
    dt: float,
    max_episode_length: int,
) -> None:
    """Require one stable profile for every episode used by a strict gate."""
    if not require_all_profiles:
        return
    if float(dt) <= 0.0:
        raise ValueError("environment dt must be positive")
    resampling_steps = int(float(resampling_time_s) / float(dt))
    if resampling_steps <= int(max_episode_length):
        raise ValueError(
            "strict staged profile gates require command resampling to be "
            "longer than one episode"
        )


def staged_disturb_window_ready(
    *,
    episode_count: int,
    min_episodes: int,
    profile_stats: Optional[ProfileStats] = None,
    require_all_profiles: bool = False,
) -> bool:
    """Return whether the aggregate and requested profile windows have enough data."""
    if int(episode_count) < int(min_episodes):
        return False
    if not require_all_profiles:
        return True
    if not profile_stats:
        return False
    return all(
        int(stats["episode_count"]) >= int(min_episodes)
        for stats in profile_stats.values()
    )


def _stats_pass(
    stats: WindowStats,
    *,
    min_task_return: float,
    max_fall_rate: float,
) -> bool:
    episode_count = max(int(stats["episode_count"]), 1)
    avg_task_return = float(stats["return_sum"]) / episode_count
    fall_rate = float(stats["fall_sum"]) / episode_count
    return avg_task_return >= float(min_task_return) and fall_rate <= float(max_fall_rate)


def staged_disturb_window_passes(
    *,
    episode_count: int,
    return_sum: float,
    fall_sum: float,
    min_episodes: int,
    min_task_return: float,
    max_fall_rate: float,
    profile_stats: Optional[ProfileStats] = None,
    require_all_profiles: bool = False,
) -> bool:
    """Apply aggregate gates and, when requested, the same gates per profile."""
    if not staged_disturb_window_ready(
        episode_count=episode_count,
        min_episodes=min_episodes,
        profile_stats=profile_stats,
        require_all_profiles=require_all_profiles,
    ):
        return False

    aggregate_stats = {
        "episode_count": episode_count,
        "return_sum": return_sum,
        "fall_sum": fall_sum,
    }
    if not _stats_pass(
        aggregate_stats,
        min_task_return=min_task_return,
        max_fall_rate=max_fall_rate,
    ):
        return False
    if not require_all_profiles:
        return True

    # Task-wise competence gates prevent a strong profile from hiding a weak
    # one, following competence-based automatic curricula surveyed by Portelas
    # et al. (IJCAI 2020) and the boundary adjustment principle of ADR.
    return all(
        _stats_pass(
            stats,
            min_task_return=min_task_return,
            max_fall_rate=max_fall_rate,
        )
        for stats in profile_stats.values()
    )
