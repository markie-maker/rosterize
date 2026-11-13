from __future__ import annotations

import random
import statistics
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
    return _number(player.get("fppg")) or 0.0


def _games(player: Mapping[str, Any], matchup_week: Any = None) -> float:
    for key in ("games_remaining", "remaining_games", "games_this_week"):
        value = _number(player.get(key))
        if value is not None:
            return max(0.0, value)
    schedule = player.get("schedule")
    if isinstance(schedule, Mapping) and matchup_week is not None:
        for key in (str(matchup_week), f"week_{matchup_week}"):
            value = schedule.get(key)
            if isinstance(value, Mapping):
                value = value.get("games", value.get("game_count"))
            if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
                return float(len(value))
            number = _number(value)
            if number is not None:
                return max(0.0, number)
    return 1.0


def _stddev(player: Mapping[str, Any], projection: float) -> tuple[float, bool]:
    for key in ("historical_stddev", "point_stddev", "stddev"):
        value = _number(player.get(key))
        if value is not None and value >= 0:
            return value, False
    logs = player.get("game_logs") or player.get("recent_games") or []
    points = []
    if isinstance(logs, Sequence):
        for log in logs:
            if isinstance(log, Mapping):
                value = _number(
                    log.get("actual_fantasy_points", log.get("fantasy_points", log.get("fpts")))
                )
                if value is not None:
                    points.append(value)
    if len(points) > 1:
        return statistics.stdev(points), False
    return max(1.0, abs(projection) * 0.25), True


def _team_distribution(
    players: Sequence[Mapping[str, Any]],
    matchup_week: Any = None,
) -> tuple[float, float, bool]:
    mean = 0.0
    variance = 0.0
    estimated_variance = False
    for player in players:
        projection = _projection(player)
        games = _games(player, matchup_week)
        weekly_projection = _number(player.get("adjusted_weekly_projection"))
        stddev, estimated = _stddev(player, projection)
        mean += weekly_projection if weekly_projection is not None else projection * games
        variance += stddev**2 * games
        estimated_variance = estimated_variance or estimated
    return mean, variance**0.5, estimated_variance


def simulate_matchup(
    user_players: Sequence[Mapping[str, Any]],
    opponent_players: Sequence[Mapping[str, Any]],
    iterations: int = 1000,
    seed: int | None = None,
    user_current_points: float = 0.0,
    opponent_current_points: float = 0.0,
    matchup_week: Any = None,
) -> dict[str, Any]:
    if iterations < 1:
        raise ValueError("iterations must be at least 1")
    if not user_players or not opponent_players:
        raise ValueError("Both matchup teams must contain at least one player")
    user_mean, user_stddev, user_estimated = _team_distribution(user_players, matchup_week)
    opponent_mean, opponent_stddev, opponent_estimated = _team_distribution(opponent_players, matchup_week)
    rng = random.Random(seed)
    user_wins = 0.0
    user_totals: list[float] = []
    opponent_totals: list[float] = []
    for _ in range(iterations):
        user_score = user_current_points + rng.gauss(user_mean, user_stddev)
        opponent_score = opponent_current_points + rng.gauss(opponent_mean, opponent_stddev)
        user_totals.append(user_score)
        opponent_totals.append(opponent_score)
        if user_score > opponent_score:
            user_wins += 1
        elif user_score == opponent_score:
            user_wins += 0.5
    probability = user_wins / iterations
    recommendation = (
        "Aggressive Streaming Recommended"
        if 0.35 <= probability <= 0.65
        else "Hold Long-Term Stashes"
        if probability > 0.65
        else "Aggressive Streaming Recommended"
    )
    return {
        "win_probability": round(probability, 4),
        "win_probability_percent": round(probability * 100, 1),
        "iterations": iterations,
        "user_expected_points": round(user_current_points + user_mean, 2),
        "opponent_expected_points": round(opponent_current_points + opponent_mean, 2),
        "user_simulated_average": round(statistics.fmean(user_totals), 2),
        "opponent_simulated_average": round(statistics.fmean(opponent_totals), 2),
        "strategic_recommendation": recommendation,
        "variance_quality": "estimated" if user_estimated or opponent_estimated else "historical",
    }


def simulate_matchup_data(
    matchup_data: Mapping[str, Any],
    user_team_id: Any,
    iterations: int = 1000,
    seed: int | None = None,
) -> dict[str, Any]:
    user_team = matchup_data.get("user_team")
    opponent_team = matchup_data.get("opponent_team")
    if not isinstance(user_team, Mapping) or not isinstance(opponent_team, Mapping):
        raise ValueError("matchup_data must contain user_team and opponent_team")
    result = simulate_matchup(
        user_team.get("roster", []),
        opponent_team.get("roster", []),
        iterations=iterations,
        seed=seed,
        user_current_points=_number(user_team.get("current_points")) or 0.0,
        opponent_current_points=_number(opponent_team.get("current_points")) or 0.0,
        matchup_week=matchup_data.get("current_scoring_period", matchup_data.get("current_matchup_period")),
    )
    result["user_team_id"] = user_team_id
    result["opponent_team_id"] = opponent_team.get("id")
    return result
