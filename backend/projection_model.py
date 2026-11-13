from __future__ import annotations

from collections.abc import Mapping, Sequence
from os import PathLike
from typing import Any

from xgboost import XGBRegressor


MODEL_FEATURES = (
    "minutes_trend",
    "usage_trend",
    "injury_flag",
    "schedule_density",
)


def _number(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if number == number else None


def _value(player: Mapping[str, Any], *keys: str, default: float = 0.0) -> float:
    for key in keys:
        number = _number(player.get(key))
        if number is not None:
            return number
    return default


def _projection(player: Mapping[str, Any]) -> float:
    projection = _number(player.get("projection_fppg"))
    if projection is not None:
        return projection
    projection = _number(player.get("espn_projected_points"))
    if projection is not None:
        return projection
    timeframes = player.get("timeframe_stats")
    if isinstance(timeframes, Mapping) and isinstance(timeframes.get("proj"), Mapping):
        projection = _number(timeframes["proj"].get("fppg"))
        if projection is not None:
            return projection
    return _value(player, "projected_fppg", "fppg", "avg_points")


def _injury_flag(player: Mapping[str, Any]) -> float:
    status = str(player.get("injury_status", player.get("injuryStatus", ""))).upper()
    return float(bool(player.get("injured")) or status in {"OUT", "INJURY_RESERVE", "IR", "DOUBTFUL"})


def _feature_row(player: Mapping[str, Any]) -> list[float]:
    games_remaining = _number(player.get("games_remaining"))
    schedule_density = _number(player.get("schedule_density"))
    if schedule_density is None:
        schedule_density = games_remaining / 7 if games_remaining is not None else 0.0
    return [
        _value(player, "minutes_trend"),
        _value(player, "usage_trend"),
        _injury_flag(player),
        schedule_density,
    ]


def _outlook_tag(player: Mapping[str, Any], residual: float) -> str:
    minutes_trend = _number(player.get("minutes_trend"))
    usage_trend = _number(player.get("usage_trend"))
    delta = _number(player.get("projection_delta_7d"))
    recent_efficiency = _number(player.get("recent_fg_pct"))
    baseline_efficiency = _number(player.get("baseline_fg_pct"))
    if minutes_trend is not None and minutes_trend >= 0.15:
        return f"Trending Up: Minutes +{minutes_trend:.0%}"
    if (
        delta is not None
        and delta < 0
        and minutes_trend is not None
        and usage_trend is not None
        and minutes_trend >= -0.05
        and usage_trend >= -0.05
    ):
        return "Buy-Low Candidate: Production below projection with a stable role"
    if (
        delta is not None
        and delta > 0
        and recent_efficiency is not None
        and baseline_efficiency is not None
        and recent_efficiency - baseline_efficiency >= 0.08
        and (usage_trend is None or usage_trend < 0.1)
    ):
        return "Regression Candidate: Recent efficiency spike"
    if residual >= 1:
        return "Trending Up: Model correction is positive"
    if residual <= -1:
        return "Trending Down: Model correction is negative"
    return "On Track"


class ProjectionCorrectionModel:
    def __init__(
        self,
        min_training_rows: int = 20,
        random_state: int = 42,
    ) -> None:
        if min_training_rows < 2:
            raise ValueError("min_training_rows must be at least 2")
        self.min_training_rows = min_training_rows
        self.model = XGBRegressor(
            n_estimators=250,
            max_depth=3,
            learning_rate=0.04,
            subsample=0.85,
            colsample_bytree=1.0,
            reg_lambda=2.0,
            objective="reg:squarederror",
            n_jobs=1,
            random_state=random_state,
        )
        self.is_fitted = False

    def fit(self, training_rows: Sequence[Mapping[str, Any]]) -> ProjectionCorrectionModel:
        rows: list[Mapping[str, Any]] = []
        targets: list[float] = []
        for row in training_rows:
            actual = _number(row.get("actual_points", row.get("actual_fantasy_points")))
            projected = _number(row.get("espn_projected_points", row.get("projected_points")))
            if actual is None or projected is None:
                continue
            rows.append(row)
            targets.append(actual - projected)
        if len(rows) < self.min_training_rows:
            raise ValueError(
                f"At least {self.min_training_rows} labeled player-game rows are required; received {len(rows)}"
            )
        self.model.fit([_feature_row(row) for row in rows], targets)
        self.is_fitted = True
        return self

    def predict_residual(self, player: Mapping[str, Any]) -> float:
        if not self.is_fitted:
            raise RuntimeError("ProjectionCorrectionModel must be fitted before prediction")
        return float(self.model.predict([_feature_row(player)])[0])

    def adjust_player(self, player: Mapping[str, Any]) -> dict[str, Any]:
        residual = self.predict_residual(player)
        adjusted = _projection(player) + residual
        return {
            **player,
            "projection_residual": residual,
            "adjusted_projection": adjusted,
            "outlook_tag": _outlook_tag(player, residual),
        }

    def adjust_players(self, players: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
        return [self.adjust_player(player) for player in players]

    def save(self, model_path: str | PathLike[str]) -> None:
        if not self.is_fitted:
            raise RuntimeError("ProjectionCorrectionModel must be fitted before it can be saved")
        self.model.save_model(str(model_path))

    def load(self, model_path: str | PathLike[str]) -> ProjectionCorrectionModel:
        self.model.load_model(str(model_path))
        self.is_fitted = True
        return self
