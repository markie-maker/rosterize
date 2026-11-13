from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any


def _number(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if number == number else None


def _projection(player: Mapping[str, Any]) -> float:
    for key in ("adjusted_projection", "projection_fppg", "projected_fppg"):
        value = _number(player.get(key))
        if value is not None:
            return value
    timeframes = player.get("timeframe_stats")
    if isinstance(timeframes, Mapping) and isinstance(timeframes.get("proj"), Mapping):
        value = _number(timeframes["proj"].get("fppg"))
        if value is not None:
            return value
    value = _number(player.get("fppg"))
    if value is not None:
        return value
    return 0.0


def _remaining_games(player: Mapping[str, Any], matchup_week: Any = None) -> float:
    for key in ("remaining_games", "games_remaining", "games_this_week"):
        value = _number(player.get(key))
        if value is not None:
            return max(0.0, value)
    schedule = player.get("schedule")
    if isinstance(schedule, Mapping):
        keys = (str(matchup_week), f"week_{matchup_week}") if matchup_week is not None else ()
        for key in keys:
            value = schedule.get(key)
            if isinstance(value, Mapping):
                value = value.get("games", value.get("game_count"))
            if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
                return float(len(value))
            number = _number(value)
            if number is not None:
                return max(0.0, number)
    return 1.0


def _weekly_projection(player: Mapping[str, Any], matchup_week: Any = None) -> float:
    weekly = _number(player.get("adjusted_weekly_projection"))
    return weekly if weekly is not None else _projection(player) * _remaining_games(player, matchup_week)


def rank_waiver_players(
    free_agents: Sequence[Mapping[str, Any]],
    droppable_players: Sequence[Mapping[str, Any]] = (),
    matchup_week: Any = None,
) -> list[dict[str, Any]]:
    droppable_value = min(
        (
            _weekly_projection(player, matchup_week)
            for player in droppable_players
        ),
        default=0.0,
    )
    ranked: list[dict[str, Any]] = []
    for player in free_agents:
        games = _remaining_games(player, matchup_week)
        weekly_projection = _weekly_projection(player, matchup_week)
        result = {
            **player,
            "remaining_games": games,
            "adjusted_weekly_projection": weekly_projection,
            "droppable_player_weekly_projection": droppable_value,
            "net_weekly_gain": weekly_projection - droppable_value,
        }
        ranked.append(result)
    return sorted(ranked, key=lambda player: player["net_weekly_gain"], reverse=True)


def find_ir_stashes(
    free_agents: Sequence[Mapping[str, Any]],
    has_open_ir_slot: bool,
    minimum_season_projection: float = 0.0,
    minimum_percent_owned: float = 0.0,
) -> list[dict[str, Any]]:
    if not has_open_ir_slot:
        return []
    candidates: list[dict[str, Any]] = []
    for player in free_agents:
        status = str(player.get("injury_status", player.get("injuryStatus", ""))).upper()
        if status != "OUT":
            continue
        projection = _number(player.get("season_projection"))
        if projection is None:
            projection = _number(player.get("projected_season_fppg"))
        if projection is None:
            timeframes = player.get("timeframe_stats")
            season = player.get("season_year")
            projection = _number(player.get("season_fppg"))
            if projection is None and isinstance(timeframes, Mapping):
                season_stats = timeframes.get(str(season)) if season else None
                if isinstance(season_stats, Mapping):
                    projection = _number(season_stats.get("fppg"))
        owned = _number(player.get("percent_owned"))
        if projection is None or owned is None:
            continue
        if projection < minimum_season_projection or owned < minimum_percent_owned:
            continue
        candidates.append(
            {
                **player,
                "season_projection": projection,
                "percent_owned": owned,
                "ir_stash_score": projection * owned / 100,
                "recommendation": "IR stash candidate",
            }
        )
    return sorted(candidates, key=lambda player: player["ir_stash_score"], reverse=True)
