"""Feature engineering for FIFA player talent classification."""

from __future__ import annotations

import numpy as np
import pandas as pd


POSITION_GROUPS = {
    "GK": "GK",
    "CB": "DEF",
    "LB": "DEF",
    "LWB": "DEF",
    "RB": "DEF",
    "RWB": "DEF",
    "CDM": "MID",
    "CM": "MID",
    "CAM": "MID",
    "LM": "MID",
    "RM": "MID",
    "LW": "ATT",
    "RW": "ATT",
    "CF": "ATT",
    "ST": "ATT",
}


def create_target(df: pd.DataFrame) -> pd.Series:
    """Return the proposal's binary Top Talent target."""
    return ((df["overall"] >= 80) | (df["potential"] >= 85)).astype(int)


def primary_position(position_value: object) -> str:
    if pd.isna(position_value):
        return "UNKNOWN"
    return str(position_value).split(",")[0].strip()


def position_group(position_value: object) -> str:
    return POSITION_GROUPS.get(primary_position(position_value), "UNKNOWN")


def _mean_existing(df: pd.DataFrame, columns: list[str]) -> pd.Series:
    existing = [column for column in columns if column in df.columns]
    if not existing:
        return pd.Series(np.nan, index=df.index)
    return df[existing].mean(axis=1)


def add_custom_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add proposal-specific scouting features."""
    result = df.copy()
    result["top_talent"] = create_target(result)
    result["primary_position"] = result["player_positions"].map(primary_position)
    result["position_group"] = result["player_positions"].map(position_group)

    attack_score = _mean_existing(
        result,
        [
            "attacking_finishing",
            "attacking_crossing",
            "attacking_heading_accuracy",
            "attacking_short_passing",
            "attacking_volleys",
            "power_shot_power",
            "power_long_shots",
        ],
    )
    skill_score = _mean_existing(
        result,
        [
            "skill_dribbling",
            "skill_curve",
            "skill_fk_accuracy",
            "skill_long_passing",
            "skill_ball_control",
        ],
    )
    movement_score = _mean_existing(
        result,
        [
            "movement_acceleration",
            "movement_sprint_speed",
            "movement_agility",
            "movement_reactions",
            "movement_balance",
        ],
    )
    defense_score = _mean_existing(
        result,
        [
            "defending_marking_awareness",
            "defending_standing_tackle",
            "defending_sliding_tackle",
            "mentality_interceptions",
            "mentality_aggression",
        ],
    )
    gk_score = _mean_existing(
        result,
        [
            "goalkeeping_diving",
            "goalkeeping_handling",
            "goalkeeping_kicking",
            "goalkeeping_positioning",
            "goalkeeping_reflexes",
        ],
    )

    result["performance_index"] = np.select(
        [
            result["position_group"].eq("GK"),
            result["position_group"].eq("DEF"),
            result["position_group"].eq("MID"),
            result["position_group"].eq("ATT"),
        ],
        [
            gk_score,
            0.55 * defense_score + 0.20 * movement_score + 0.15 * skill_score + 0.10 * attack_score,
            0.40 * skill_score + 0.30 * movement_score + 0.20 * attack_score + 0.10 * defense_score,
            0.45 * attack_score + 0.30 * skill_score + 0.20 * movement_score + 0.05 * defense_score,
        ],
        default=_mean_existing(result, ["overall", "potential"]),
    )

    result["age_adjusted_potential"] = (result["potential"] - result["overall"]) / result["age"].clip(lower=1)
    value_denominator = np.log1p(result["value_eur"].fillna(0).clip(lower=0)).replace(0, np.nan)
    result["market_value_efficiency"] = result["performance_index"] / value_denominator
    result["market_value_efficiency"] = result["market_value_efficiency"].replace([np.inf, -np.inf], np.nan)
    result["wage_value_ratio"] = result["wage_eur"].fillna(0) / result["value_eur"].replace(0, np.nan)
    result["wage_value_ratio"] = result["wage_value_ratio"].replace([np.inf, -np.inf], np.nan).fillna(0)

    return result
