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
    for key in ("fppg", "avg_points"):
        value = _number(player.get(key))
        if value is not None:
            return value
    return 0.0


def patience_score(player: Mapping[str, Any]) -> int:
    score = 50.0
    owned = _number(player.get("percent_owned"))
    if owned is not None:
        score += (max(0.0, min(100.0, owned)) - 50) * 0.25
    draft_pick = _number(player.get("draft_pick", player.get("draft_position")))
    if draft_pick is not None and draft_pick > 0:
        score += max(-5.0, min(15.0, (120 - draft_pick) / 8))
    injury = str(player.get("injury_status", player.get("injuryStatus", ""))).upper()
    if injury in {"OUT", "IR", "INJURY_RESERVE"}:
        score -= 18
    elif injury in {"DOUBTFUL", "QUESTIONABLE"}:
        score -= 8
    elif injury in {"HEALTHY", "ACTIVE", "PROBABLE"}:
        score += 4
    minutes = _number(player.get("minutes_trend"))
    usage = _number(player.get("usage_trend"))
    if minutes is not None:
        score += max(-15.0, min(15.0, minutes * 40))
    if usage is not None:
        score += max(-10.0, min(10.0, usage * 25))
    if player.get("injured") is True and injury not in {"OUT", "IR", "INJURY_RESERVE"}:
        score -= 8
    return int(round(max(1.0, min(100.0, score))))


def analyze_player_market(player: Mapping[str, Any]) -> dict[str, Any]:
    result = dict(player)
    delta = _number(player.get("projection_delta_7d"))
    if delta is None:
        delta = _number(player.get("projection_delta_30d"))
    minutes = _number(player.get("minutes_trend"))
    usage = _number(player.get("usage_trend"))
    recent_efficiency = _number(player.get("recent_fg_pct"))
    baseline_efficiency = _number(player.get("baseline_fg_pct"))
    efficiency_spike = (
        recent_efficiency is not None
        and baseline_efficiency is not None
        and recent_efficiency - baseline_efficiency >= 0.08
        and (usage is None or usage < 0.1)
    )
    teammate_injury = bool(
        player.get("benefits_from_teammate_injury")
        or (_number(player.get("injured_teammates")) or 0) > 0
    )
    buy_low = (
        delta is not None
        and delta < 0
        and minutes is not None
        and usage is not None
        and minutes >= -0.05
        and usage >= -0.05
    )
    sell_high = delta is not None and delta > 0 and (efficiency_spike or teammate_injury)
    if buy_low:
        tag = "BUY_LOW"
        rationale = "Production is below projection while minutes and shot attempts remain stable or rise."
    elif sell_high:
        tag = "SELL_HIGH"
        reason = "efficiency spike" if efficiency_spike else "teammate injury opportunity"
        rationale = f"Recent outperformance may rely on a temporary {reason}."
    elif delta is not None and delta < 0:
        tag = "UNDERPERFORMER"
        rationale = "Recent production is below ESPN projection."
    elif delta is not None and delta > 0:
        tag = "OVERPERFORMER"
        rationale = "Recent production is above ESPN projection."
    else:
        tag = "NEUTRAL"
        rationale = "Not enough recent projection data for a market signal."
    result.update(
        {
            "market_tag": tag,
            "market_rationale": rationale,
            "patience_score": patience_score(player),
        }
    )
    return result


def analyze_market(players: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    return sorted(
        (analyze_player_market(player) for player in players),
        key=lambda player: (
            player["market_tag"] in {"BUY_LOW", "SELL_HIGH", "UNDERPERFORMER", "OVERPERFORMER"},
            abs(_number(player.get("projection_delta_7d")) or 0.0),
        ),
        reverse=True,
    )
