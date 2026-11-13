from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
from itertools import combinations
from typing import Any

from data_pipeline import build_player_features


POSITIONS = ("PG", "SG", "SF", "PF", "C")
_IGNORED_POSITIONS = {"G", "F", "UTIL", "UT", "BE", "IR", "IL"}


def _number(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if number == number else None


def _players(team: Mapping[str, Any]) -> list[dict[str, Any]]:
    roster = team.get("roster", [])
    return [dict(player) for player in roster if isinstance(player, Mapping)]


def _teams(data: Mapping[str, Any]) -> list[dict[str, Any]]:
    teams = data.get("teams")
    if isinstance(teams, Sequence) and not isinstance(teams, (str, bytes)):
        return [dict(team) for team in teams if isinstance(team, Mapping)]
    return [
        dict(team)
        for key in ("user_team", "target_team")
        if isinstance((team := data.get(key)), Mapping)
    ]


def _player_projection(player: Mapping[str, Any]) -> float:
    direct = _number(player.get("adjusted_weekly_projection"))
    if direct is not None:
        return direct
    for key in ("adjusted_projection", "projection_fppg", "projected_fppg", "espn_projected_points"):
        direct = _number(player.get(key))
        if direct is not None:
            return direct
    projection = player.get("projection")
    if isinstance(projection, Mapping):
        direct = _number(projection.get("fppg") or projection.get("points"))
        if direct is not None:
            return direct
    timeframes = player.get("timeframe_stats")
    if isinstance(timeframes, Mapping) and isinstance(timeframes.get("proj"), Mapping):
        direct = _number(timeframes["proj"].get("fppg"))
        if direct is not None:
            return direct
    return _number(player.get("fppg") or player.get("avg_points")) or 0.0


def _positions(player: Mapping[str, Any]) -> tuple[str, ...]:
    result: set[str] = set()
    for key in ("positions", "position", "eligible_slots"):
        value = player.get(key)
        values = value.replace(",", "/").split("/") if isinstance(value, str) else value
        if isinstance(values, Sequence) and not isinstance(values, (str, bytes)):
            for position in values:
                result.update(
                    part.strip().upper()
                    for part in str(position).replace(",", "/").split("/")
                )
    return tuple(position for position in POSITIONS if position in result)


def _player_identifier(player: Mapping[str, Any]) -> Any:
    return player.get("id", player.get("player_id", player.get("playerId")))


def _resolve_players(team: Mapping[str, Any], selectors: Sequence[Any]) -> list[dict[str, Any]]:
    roster = _players(team)
    resolved: list[dict[str, Any]] = []
    for selector in selectors:
        if isinstance(selector, Mapping):
            selector_id = _player_identifier(selector)
            selector_name = selector.get("name")
        elif isinstance(selector, str) and selector.strip().isdigit():
            selector_id = int(selector.strip())
            selector_name = None
        else:
            selector_id = selector if isinstance(selector, (int, float)) else None
            selector_name = str(selector) if selector_id is None else None
        match = next(
            (
                player
                for player in roster
                if (selector_id is not None and str(_player_identifier(player)) == str(selector_id))
                or (
                    selector_name is not None
                    and str(player.get("name", "")).casefold() == selector_name.casefold()
                )
            ),
            None,
        )
        if match is None:
            raise ValueError(f"Could not find trade player {selector!r} on team {team.get('name', team.get('id'))!r}")
        if match in resolved:
            raise ValueError(f"Trade player {selector!r} was selected more than once")
        resolved.append(match)
    return resolved


def _playoff_games(player: Mapping[str, Any]) -> float | None:
    explicit = _number(player.get("playoff_games"))
    if explicit is not None:
        return explicit
    schedule = player.get("schedule")
    if not isinstance(schedule, Mapping):
        return None
    total = 0.0
    found = False
    for week in ("20", "21", "22"):
        value = schedule.get(week, schedule.get(f"week_{week}"))
        if isinstance(value, Mapping):
            value = value.get("games", value.get("game_count"))
        if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
            value = len(value)
        games = _number(value)
        if games is not None:
            total += games
            found = True
    return total if found else None


def _schedule_adjustment(player: Mapping[str, Any], projection: float) -> float:
    games = _playoff_games(player)
    if games is None:
        return 0.0
    return projection * (games / 9 - 1)


def _explicit_needs(team: Mapping[str, Any]) -> dict[str, float]:
    raw = team.get("position_needs")
    if not isinstance(raw, Mapping):
        return {}
    return {str(key).upper(): max(0.0, _number(value) or 0.0) for key, value in raw.items()}


def _position_counts(team: Mapping[str, Any]) -> Counter[str]:
    counts: Counter[str] = Counter()
    for player in _players(team):
        for position in _positions(player):
            counts[position] += 1
    return counts


def _need_levels(team: Mapping[str, Any], teams: Sequence[Mapping[str, Any]]) -> dict[str, float]:
    explicit = _explicit_needs(team)
    if explicit:
        return explicit
    if not teams:
        return {}
    team_counts = _position_counts(team)
    all_counts = [_position_counts(candidate) for candidate in teams]
    return {
        position: max(
            0.0,
            sum(counts[position] for counts in all_counts) / len(all_counts) - team_counts[position],
        )
        for position in POSITIONS
    }


def _scarcity_adjustment(player: Mapping[str, Any]) -> float:
    explicit = _number(player.get("positional_scarcity_adjustment"))
    if explicit is not None:
        return explicit
    scarcity = _number(player.get("positional_scarcity"))
    return scarcity if scarcity is not None else 0.0


def _team_player_value(
    player: Mapping[str, Any],
    team: Mapping[str, Any],
    teams: Sequence[Mapping[str, Any]],
) -> float:
    projection = _player_projection(player)
    schedule_adjustment = _schedule_adjustment(player, projection)
    scarcity = _scarcity_adjustment(player)
    needs = _need_levels(team, teams)
    need_bonus = projection * 0.15 * max(
        (min(1.0, needs.get(position, 0.0)) for position in _positions(player)),
        default=0.0,
    )
    return projection + schedule_adjustment + scarcity + need_bonus


def _waiver_candidates(data: Mapping[str, Any]) -> list[dict[str, Any]]:
    candidates = data.get("top_free_agents") or data.get("free_agents") or []
    if isinstance(candidates, Mapping):
        candidates = candidates.get("players", [])
    if not isinstance(candidates, Sequence) or isinstance(candidates, (str, bytes)):
        return []
    return [dict(player) for player in candidates if isinstance(player, Mapping)]


def _enrich_trade_data(data: Mapping[str, Any]) -> dict[str, Any]:
    teams = _teams(data)
    free_agents = _waiver_candidates(data)
    players = [player for team in teams for player in _players(team)] + free_agents
    features = build_player_features(players, free_agents=free_agents)
    by_id: dict[str, dict[str, Any]] = {}
    by_name: dict[str, dict[str, Any]] = {}
    for player in features:
        player_id = _player_identifier(player)
        if player_id is not None:
            by_id[str(player_id)] = player
        if player.get("name"):
            by_name[str(player["name"]).casefold()] = player

    def enriched_player(player: Mapping[str, Any]) -> dict[str, Any]:
        player_id = _player_identifier(player)
        if player_id is not None and str(player_id) in by_id:
            return by_id[str(player_id)]
        return by_name.get(str(player.get("name", "")).casefold(), dict(player))

    enriched_teams = [
        {
            **team,
            "roster": [enriched_player(player) for player in _players(team)],
        }
        for team in teams
    ]
    result = {
        **data,
        "teams": enriched_teams,
        "top_free_agents": [enriched_player(player) for player in free_agents],
        "_features_enriched": True,
    }
    for key in ("user_team", "target_team"):
        original = data.get(key)
        if not isinstance(original, Mapping):
            continue
        team_id = original.get("id")
        replacement = next(
            (
                team
                for team in enriched_teams
                if team.get("id") == team_id
                or (team_id is None and team.get("name") == original.get("name"))
            ),
            None,
        )
        result[key] = replacement or {
            **original,
            "roster": [enriched_player(player) for player in _players(original)],
        }
    return result


def _best_waiver(data: Mapping[str, Any], waiver_player: Any = None) -> dict[str, Any] | None:
    candidates = _waiver_candidates(data)
    if isinstance(waiver_player, Mapping):
        return dict(waiver_player)
    if waiver_player is not None:
        selected = _resolve_players({"name": "free agent", "roster": candidates}, [waiver_player])
        return selected[0]
    return max(candidates, key=_player_projection) if candidates else None


def _grade(user_gain: float, outgoing_value: float) -> str:
    ratio = user_gain / max(1.0, outgoing_value)
    if ratio >= 0.25:
        return "A"
    if ratio >= 0.15:
        return "B"
    if ratio >= 0.05:
        return "C"
    if ratio > 0:
        return "D"
    return "F"


def _team_gain(
    team: Mapping[str, Any],
    incoming: Sequence[Mapping[str, Any]],
    outgoing: Sequence[Mapping[str, Any]],
    teams: Sequence[Mapping[str, Any]],
) -> tuple[float, dict[str, float]]:
    incoming_value = sum(_team_player_value(player, team, teams) for player in incoming)
    outgoing_value = sum(_team_player_value(player, team, teams) for player in outgoing)
    projected = sum(_player_projection(player) for player in incoming) - sum(
        _player_projection(player) for player in outgoing
    )
    need_improvement: Counter[str] = Counter()
    for player in incoming:
        for position in _positions(player):
            need_improvement[position] += 1
    for player in outgoing:
        for position in _positions(player):
            need_improvement[position] -= 1
    needs = _need_levels(team, teams)
    resolved = {
        position: min(need, max(0.0, need_improvement[position]))
        for position, need in needs.items()
        if need > 0
    }
    return incoming_value - outgoing_value, {
        "projected_points": projected,
        "positional_need_improvement": sum(resolved.values()),
        "playoff_schedule_adjustment": sum(_schedule_adjustment(player, _player_projection(player)) for player in incoming)
        - sum(_schedule_adjustment(player, _player_projection(player)) for player in outgoing),
        "positional_scarcity_adjustment": sum(_scarcity_adjustment(player) for player in incoming)
        - sum(_scarcity_adjustment(player) for player in outgoing),
    }


def _pitch(team: Mapping[str, Any], incoming: Sequence[Mapping[str, Any]], outgoing: Sequence[Mapping[str, Any]], gain: float) -> list[str]:
    positions = sorted({position for player in incoming for position in _positions(player)})
    name = team.get("name", "Your team")
    pitch = [f"{name}: {gain:+.1f} projected points per game after the exchange."]
    if positions:
        pitch.append(f"Adds depth at {', '.join(positions)} with {', '.join(str(player.get('name', 'player')) for player in incoming)}.")
    if outgoing:
        pitch.append(f"Consolidates {', '.join(str(player.get('name', 'player')) for player in outgoing)} into a better roster fit.")
    return pitch


def evaluate_trade(
    trade_data: Mapping[str, Any],
    user_gives: Sequence[Any],
    target_gives: Sequence[Any],
    waiver_player: Any = None,
) -> dict[str, Any]:
    if not trade_data.get("_features_enriched"):
        trade_data = _enrich_trade_data(trade_data)
    if not user_gives or not target_gives:
        raise ValueError("Both teams must give at least one player")
    user_team = trade_data.get("user_team")
    target_team = trade_data.get("target_team")
    if not isinstance(user_team, Mapping) or not isinstance(target_team, Mapping):
        raise ValueError("trade_data must contain user_team and target_team objects")
    user_outgoing = _resolve_players(user_team, user_gives)
    target_outgoing = _resolve_players(target_team, target_gives)
    teams = _teams(trade_data)
    waiver = None
    user_incoming = list(target_outgoing)
    target_incoming = list(user_outgoing)
    waiver_team = None
    if len(target_gives) < len(user_gives):
        waiver = _best_waiver(trade_data, waiver_player)
        if waiver is not None:
            user_incoming.append(waiver)
            waiver_team = "user"
    elif len(user_gives) < len(target_gives):
        waiver = _best_waiver(trade_data, waiver_player)
        if waiver is not None:
            target_incoming.append(waiver)
            waiver_team = "target"

    user_gain, user_breakdown = _team_gain(user_team, user_incoming, user_outgoing, teams)
    target_gain, target_breakdown = _team_gain(target_team, target_incoming, target_outgoing, teams)
    user_outgoing_value = sum(_player_projection(player) for player in user_outgoing)
    recommendation = "TRADE" if user_gain > 0 and target_gain > 0 else "STAND_PAT"
    rationale = (
        f"Projected value change: {user_team.get('name', 'User')} {user_gain:+.1f}, "
        f"{target_team.get('name', 'Target')} {target_gain:+.1f} points per game."
    )
    if waiver is not None:
        rationale += f" {waiver.get('name', 'A free agent')} is included for the team receiving fewer players."
    return {
        "recommendation_type": recommendation,
        "trade_grade": _grade(user_gain, user_outgoing_value),
        "trade_details": {
            "user_gives": [str(player.get("name", _player_identifier(player))) for player in user_outgoing],
            "target_gives": [str(player.get("name", _player_identifier(player))) for player in target_outgoing],
            "waiver_add": str(waiver.get("name")) if waiver is not None else None,
            "waiver_team": waiver_team,
        },
        "user_team": {
            "id": user_team.get("id"),
            "name": user_team.get("name"),
            "net_projected_points": round(user_gain, 2),
            **{key: round(value, 2) for key, value in user_breakdown.items()},
            "pitch": _pitch(user_team, user_incoming, user_outgoing, user_gain),
        },
        "target_team": {
            "id": target_team.get("id"),
            "name": target_team.get("name"),
            "net_projected_points": round(target_gain, 2),
            **{key: round(value, 2) for key, value in target_breakdown.items()},
            "pitch": _pitch(target_team, target_incoming, target_outgoing, target_gain),
        },
        "strategic_rationale": rationale,
    }


def find_trades(
    league_data: Mapping[str, Any],
    user_team_id: Any,
    target_team_id: Any = None,
    max_results: int = 10,
) -> list[dict[str, Any]]:
    if max_results < 1:
        raise ValueError("max_results must be at least 1")
    if not league_data.get("_features_enriched"):
        league_data = _enrich_trade_data(league_data)
    teams = _teams(league_data)
    user_team = next((team for team in teams if team.get("id") == user_team_id), None)
    if user_team is None:
        raise ValueError(f"Could not find user team {user_team_id!r}")
    opponents = [
        team
        for team in teams
        if team.get("id") != user_team_id
        and (target_team_id is None or team.get("id") == target_team_id)
    ]
    if target_team_id is not None and not opponents:
        raise ValueError(f"Could not find target team {target_team_id!r}")
    free_agents = _waiver_candidates(league_data)
    ranked: list[dict[str, Any]] = []
    for target_team in opponents:
        trade_data = {
            **league_data,
            "user_team": user_team,
            "target_team": target_team,
            "teams": teams,
            "top_free_agents": free_agents,
        }
        user_roster = _players(user_team)
        target_roster = _players(target_team)
        patterns = ((1, 1), (2, 1), (1, 2))
        for user_count, target_count in patterns:
            if len(user_roster) < user_count or len(target_roster) < target_count:
                continue
            for user_group in combinations(user_roster, user_count):
                for target_group in combinations(target_roster, target_count):
                    result = evaluate_trade(
                        trade_data,
                        [_player_identifier(player) if _player_identifier(player) is not None else player["name"] for player in user_group],
                        [_player_identifier(player) if _player_identifier(player) is not None else player["name"] for player in target_group],
                    )
                    user_gain = result["user_team"]["net_projected_points"]
                    target_gain = result["target_team"]["net_projected_points"]
                    both_gain = user_gain > 0 and target_gain > 0
                    both_address_needs = (
                        result["user_team"]["positional_need_improvement"] > 0
                        and result["target_team"]["positional_need_improvement"] > 0
                    )
                    if not (both_gain or both_address_needs):
                        continue
                    result["trade_details"]["user_gives"] = [dict(player) for player in user_group]
                    result["trade_details"]["target_gives"] = [dict(player) for player in target_group]
                    result["team_ids"] = {
                        "user": user_team.get("id"),
                        "target": target_team.get("id"),
                    }
                    result["trade_pitch"] = {
                        "user": result["user_team"]["pitch"],
                        "target": result["target_team"]["pitch"],
                    }
                    ranked.append(result)
    ranked.sort(
        key=lambda result: (
            min(
                result["user_team"]["net_projected_points"],
                result["target_team"]["net_projected_points"],
            ),
            result["user_team"]["net_projected_points"] + result["target_team"]["net_projected_points"],
        ),
        reverse=True,
    )
    return ranked[:max_results]


def find_trades_between_teams(
    trade_data: Mapping[str, Any],
    user_team_id: Any,
    target_team_id: Any,
    max_results: int = 10,
) -> list[dict[str, Any]]:
    data = dict(trade_data)
    data["teams"] = _teams(trade_data)
    return find_trades(data, user_team_id, target_team_id, max_results)
