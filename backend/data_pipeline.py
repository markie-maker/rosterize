from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping
from datetime import date, datetime
from typing import Any


POSITIONS = ("PG", "SG", "SF", "PF", "C")
_PLAYER_KEYS = {"id", "player_id", "playerId", "name"}


def _as_records(payload: Any) -> list[dict[str, Any]]:
    if payload is None:
        return []
    if hasattr(payload, "to_dict") and not isinstance(payload, Mapping):
        try:
            records = payload.to_dict(orient="records")
        except TypeError:
            records = None
        if isinstance(records, list):
            return [dict(record) for record in records if isinstance(record, Mapping)]
    if isinstance(payload, Mapping):
        for key in ("players", "roster", "free_agents", "top_free_agents", "data", "results"):
            nested = payload.get(key)
            if isinstance(nested, (list, tuple)):
                return _as_records(nested)
        if _PLAYER_KEYS.intersection(payload):
            return [dict(payload)]
        return []
    if isinstance(payload, (list, tuple)):
        return [dict(player) for player in payload if isinstance(player, Mapping)]
    return []


def _number(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if number == number else None


def _first_number(source: Mapping[str, Any], keys: tuple[str, ...]) -> float | None:
    for key in keys:
        value = _number(source.get(key))
        if value is not None:
            return value
    return None


def _timeframe_stats(player: Mapping[str, Any]) -> Mapping[str, Any]:
    timeframes = player.get("timeframe_stats")
    return timeframes if isinstance(timeframes, Mapping) else {}


def _stat_average(block: Any, keys: tuple[str, ...]) -> float | None:
    if not isinstance(block, Mapping):
        return None
    for key in keys:
        value = _number(block.get(key))
        if value is not None:
            return value
    averages = block.get("avg")
    if isinstance(averages, Mapping):
        for key in keys:
            value = _number(averages.get(key))
            if value is not None:
                return value
    return None


def _projected_fppg(player: Mapping[str, Any]) -> float | None:
    timeframes = _timeframe_stats(player)
    projection_block = timeframes.get("proj")
    projected = _stat_average(projection_block, ("fppg", "applied_avg", "projected_fppg"))
    if projected is not None:
        return projected
    for key, block in (player.get("stats") or {}).items() if isinstance(player.get("stats"), Mapping) else ():
        if str(key).endswith("_projected") and isinstance(block, Mapping):
            projected = _stat_average(block, ("applied_avg", "fppg", "projected_fppg"))
            if projected is not None:
                return projected
    return _first_number(
        player,
        ("adjusted_projection", "projected_fppg", "espn_projected_points", "fppg", "avg_points"),
    )


def _positions(player: Mapping[str, Any]) -> tuple[str, ...]:
    values: list[str] = []
    for key in ("positions", "position", "eligible_slots"):
        raw = player.get(key)
        if isinstance(raw, str):
            values.extend(raw.replace(",", "/").split("/"))
        elif isinstance(raw, (list, tuple, set)):
            for value in raw:
                values.extend(str(value).replace(",", "/").split("/"))
    normalized = {value.strip().upper() for value in values}
    return tuple(position for position in POSITIONS if position in normalized)


def _as_date(value: Any) -> date | None:
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    if isinstance(value, str):
        try:
            return datetime.fromisoformat(value.replace("Z", "+00:00")).date()
        except ValueError:
            try:
                return date.fromisoformat(value[:10])
            except ValueError:
                return None
    return None


def _game_logs(player: Mapping[str, Any]) -> list[dict[str, Any]]:
    for key in ("game_logs", "recent_games", "logs"):
        logs = player.get(key)
        if isinstance(logs, (list, tuple)):
            return [dict(log) for log in logs if isinstance(log, Mapping)]
    return []


def _log_value(log: Mapping[str, Any], keys: tuple[str, ...]) -> float | None:
    return _first_number(log, keys)


def _window_logs(logs: list[dict[str, Any]], as_of: date, days: int) -> list[dict[str, Any]]:
    start = as_of.toordinal() - days + 1
    return [
        log
        for log in logs
        if (game_date := _as_date(log.get("date") or log.get("game_date") or log.get("start_date")))
        and start <= game_date.toordinal() <= as_of.toordinal()
    ]


def _mean(logs: list[dict[str, Any]], keys: tuple[str, ...]) -> float | None:
    values = [value for log in logs if (value := _log_value(log, keys)) is not None]
    return sum(values) / len(values) if values else None


def _trend(recent: float | None, baseline: float | None) -> float | None:
    if recent is None or baseline is None or baseline == 0:
        return None
    return recent / baseline - 1


def _aggregate_trend(timeframes: Mapping[str, Any], keys: tuple[str, ...]) -> tuple[float | None, float | None, float | None]:
    recent = _stat_average(timeframes.get("7d"), keys)
    baseline = _stat_average(timeframes.get("30d"), keys)
    return recent, baseline, _trend(recent, baseline)


def _replacement_values(free_agents: Any, replacement_rank: int) -> dict[str, float]:
    if replacement_rank < 1:
        raise ValueError("replacement_rank must be at least 1")
    candidates: dict[str, list[float]] = defaultdict(list)
    for player in _as_records(free_agents):
        projection = _projected_fppg(player)
        if projection is None:
            continue
        for position in _positions(player):
            candidates[position].append(projection)
    return {
        position: sorted(values, reverse=True)[min(replacement_rank, len(values)) - 1]
        for position, values in candidates.items()
        if values
    }


def build_player_features(
    payload: Any,
    free_agents: Any = None,
    replacement_rank: int = 5,
    as_of: date | str | None = None,
) -> list[dict[str, Any]]:
    players = _as_records(payload)
    if free_agents is None and isinstance(payload, Mapping):
        free_agents = payload.get("top_free_agents") or payload.get("free_agents")
    replacement_values = _replacement_values(free_agents, replacement_rank)
    requested_date = _as_date(as_of)
    features: list[dict[str, Any]] = []

    for player in players:
        timeframes = _timeframe_stats(player)
        logs = _game_logs(player)
        dated_logs = [log for log in logs if _as_date(log.get("date") or log.get("game_date") or log.get("start_date"))]
        player_date = requested_date
        if player_date is None and dated_logs:
            player_date = max(
                _as_date(log.get("date") or log.get("game_date") or log.get("start_date"))
                for log in dated_logs
            )
        feature_date = player_date or date.today()
        deltas: dict[str, float | None] = {}
        for days in (7, 14, 30):
            window = _window_logs(logs, feature_date, days)
            pairs = [
                (
                    _log_value(log, ("actual_fantasy_points", "fantasy_points", "actual_points", "fpts", "applied_total")),
                    _log_value(log, ("espn_projected_points", "projected_fantasy_points", "projected_points", "projection", "projected_fpts")),
                )
                for log in window
            ]
            valid_pairs = [(actual, projected) for actual, projected in pairs if actual is not None and projected is not None]
            deltas[f"projection_delta_{days}d"] = (
                sum(actual - projected for actual, projected in valid_pairs) if valid_pairs else None
            )

        if logs:
            recent = _window_logs(logs, feature_date, 7)
            prior = [
                log
                for log in _window_logs(logs, feature_date, 30)
                if 7 < (feature_date - (_as_date(log.get("date") or log.get("game_date") or log.get("start_date")))).days
            ]
            minutes_recent = _mean(recent, ("minutes", "min", "MPG"))
            minutes_baseline = _mean(prior, ("minutes", "min", "MPG"))
            attempts_recent = _mean(recent, ("shot_attempts", "fga", "field_goal_attempts"))
            attempts_baseline = _mean(prior, ("shot_attempts", "fga", "field_goal_attempts"))
            minutes_trend = _trend(minutes_recent, minutes_baseline)
            usage_trend = _trend(attempts_recent, attempts_baseline)
        else:
            minutes_recent, minutes_baseline, minutes_trend = _aggregate_trend(timeframes, ("min", "minutes", "MPG", "MIN"))
            attempts_recent, attempts_baseline, usage_trend = _aggregate_trend(timeframes, ("fga", "shot_attempts", "FGA"))

        positions = _positions(player)
        player_projection = _projected_fppg(player)
        position_values = {
            position: replacement_values[position]
            for position in positions
            if position in replacement_values
        }
        best_replacement = max(position_values.values()) if position_values else None
        positional_scarcity = (
            player_projection - best_replacement
            if player_projection is not None and best_replacement is not None
            else None
        )
        features.append(
            {
                **player,
                **deltas,
                "projection_fppg": player_projection,
                "recent_minutes_avg": minutes_recent,
                "baseline_minutes_avg": minutes_baseline,
                "minutes_trend": minutes_trend,
                "minutes_trend_source": "game_logs" if logs else "timeframe_stats_7d_vs_30d",
                "recent_shot_attempts_avg": attempts_recent,
                "baseline_shot_attempts_avg": attempts_baseline,
                "usage_trend": usage_trend,
                "usage_trend_basis": "shot_attempts",
                "usage_trend_source": "game_logs" if logs else "timeframe_stats_7d_vs_30d",
                "positional_replacement_values": position_values,
                "positional_scarcity": positional_scarcity,
                "replacement_value_by_position": replacement_values,
            }
        )
    return features


def process_espn_payload(
    payload: Any,
    free_agents: Any = None,
    replacement_rank: int = 5,
    as_of: date | str | None = None,
) -> list[dict[str, Any]]:
    return build_player_features(payload, free_agents, replacement_rank, as_of)
