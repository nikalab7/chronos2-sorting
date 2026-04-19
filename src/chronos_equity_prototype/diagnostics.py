from __future__ import annotations

from math import ceil
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from .backtest import BacktestRunner, _annualized_sharpe
from .portfolio import PortfolioBuilder


EXPECTED_DATA_QUALITY_COLUMNS = [
    "basket_id",
    "prediction_session",
    "ticker",
    "slot_id",
    "event_type",
    "reason",
    "action_taken",
]

STRICT_EXECUTION_MODE = "strict"
STRICT_EXECUTION_MODE_LABEL = "strict_near_open_execution_benchmark"
EXACT_OPEN_ENTRY_FORMULA = "first_09_30_5m_bar_open"
FIRST_5M_ENTRY_FORMULA = "ohlc4_first_09_30_5m_bar"
FIRST_15M_ENTRY_FORMULA = "volume_weighted_ohlc4_first_three_5m_bars"
EXECUTION_EXIT_FORMULA = "close_of_final_regular_hours_15_55_5m_bar"
EXECUTION_ALPHA_FORMULA = "log(stock_exit/stock_entry) - log(spy_exit/spy_entry)"
EXECUTION_ENTRY_FORMULAS = {
    "exact_open": EXACT_OPEN_ENTRY_FORMULA,
    "first_5m_avg": FIRST_5M_ENTRY_FORMULA,
    "first_15m_avg": FIRST_15M_ENTRY_FORMULA,
}


def _normalize_time_columns(frame: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    normalized = frame.copy()
    for column in columns:
        if column in normalized.columns:
            normalized[column] = pd.to_datetime(normalized[column], utc=True)
    return normalized


def _build_realized_slot_paths(labeled_frame: pd.DataFrame, benchmark_ticker: str) -> pd.DataFrame:
    labeled = _normalize_time_columns(labeled_frame, ["session_date", "timestamp"])
    if "raw_slot_return" not in labeled.columns:
        raise ValueError("The labeled frame must contain `raw_slot_return` to build realized slot paths.")

    realized_slot = (
        labeled.groupby(["ticker", "session_date"], as_index=False)
        .agg(realized_slot_path=("raw_slot_return", lambda series: list(map(float, series.tolist()))))
        .rename(columns={"session_date": "prediction_session"})
    )
    spy_realized = realized_slot[realized_slot["ticker"] == benchmark_ticker][
        ["prediction_session", "realized_slot_path"]
    ].rename(columns={"realized_slot_path": "spy_realized_slot_path"})
    asset_realized = realized_slot[realized_slot["ticker"] != benchmark_ticker].merge(
        spy_realized,
        on="prediction_session",
        how="left",
        validate="many_to_one",
    )
    return asset_realized


def attach_realized_outcomes_to_forecasts(
    forecasts: pd.DataFrame,
    labeled_frame: pd.DataFrame,
    *,
    benchmark_ticker: str = "SPY",
    beta_override: float | None = None,
    recompute_scores: bool = False,
) -> pd.DataFrame:
    normalized_forecasts = _normalize_time_columns(forecasts, ["signal_session", "prediction_session"])
    realized_paths = _build_realized_slot_paths(labeled_frame, benchmark_ticker)
    frame = normalized_forecasts.merge(
        realized_paths[["ticker", "prediction_session", "realized_slot_path", "spy_realized_slot_path"]],
        on=["ticker", "prediction_session"],
        how="left",
        validate="many_to_one",
    )
    if frame["realized_slot_path"].isna().any() or frame["spy_realized_slot_path"].isna().any():
        raise ValueError("Missing realized raw slot paths while attaching realized outcomes to forecasts.")

    if beta_override is None:
        if "beta_used" in frame.columns:
            beta_used = pd.to_numeric(frame["beta_used"], errors="coerce").fillna(1.0).astype(float)
        else:
            beta_used = pd.Series(np.ones(len(frame), dtype=float), index=frame.index)
    else:
        beta_used = pd.Series(np.full(len(frame), float(beta_override), dtype=float), index=frame.index)
    frame["beta_used"] = beta_used

    if "stock_q50_path" in frame.columns and "spy_q50_path" in frame.columns:
        frame["alpha_q50_path"] = frame.apply(
            lambda row: [
                float(stock - (row["beta_used"] * spy))
                for stock, spy in zip(row["stock_q50_path"], row["spy_q50_path"])
            ],
            axis=1,
        )
    frame["realized_alpha_path"] = frame.apply(
        lambda row: [
            float(stock - (row["beta_used"] * spy))
            for stock, spy in zip(row["realized_slot_path"], row["spy_realized_slot_path"])
        ],
        axis=1,
    )
    frame["realized_day_return"] = frame["realized_alpha_path"].apply(lambda values: float(np.sum(values)))

    if recompute_scores and "alpha_q50_path" in frame.columns:
        frame["score_plain"] = frame["alpha_q50_path"].apply(lambda values: float(np.sum(values)))

        def _zscore_within_group(series: pd.Series) -> pd.Series:
            std = float(series.std(ddof=0))
            if std <= 0 or np.isnan(std):
                return pd.Series(np.zeros(len(series), dtype=float), index=series.index)
            return (series - float(series.mean())) / std

        if "basket_id" in frame.columns:
            frame["score_z"] = frame.groupby(["basket_id", "prediction_session"], group_keys=False)["score_plain"].transform(
                _zscore_within_group
            )
        if all(column in frame.columns for column in ["stock_q90_path", "stock_q10_path", "spy_q90_path", "spy_q10_path"]):
            alpha_uncertainty: list[float] = []
            for row in frame.itertuples(index=False):
                stock_width = np.array(row.stock_q90_path, dtype=float) - np.array(row.stock_q10_path, dtype=float)
                spy_width = np.array(row.spy_q90_path, dtype=float) - np.array(row.spy_q10_path, dtype=float)
                alpha_uncertainty.append(float(np.sum(stock_width + (abs(row.beta_used) * spy_width))))
            frame["score_risk"] = frame["score_plain"] / (1e-8 + np.array(alpha_uncertainty))
        if "score_z" in frame.columns:
            frame["score_q50"] = frame["score_z"]
    return frame


def apply_unit_beta_to_forecasts(
    forecasts: pd.DataFrame,
    labeled_frame: pd.DataFrame,
    *,
    benchmark_ticker: str = "SPY",
) -> pd.DataFrame:
    frame = attach_realized_outcomes_to_forecasts(
        forecasts,
        labeled_frame,
        benchmark_ticker=benchmark_ticker,
        beta_override=1.0,
        recompute_scores=True,
    )
    frame["beta_mode"] = "unit_beta_1.0"
    return frame


def _build_realized_returns(forecasts: pd.DataFrame, realized_column: str = "realized_day_return") -> pd.DataFrame:
    return (
        forecasts[["ticker", "prediction_session", realized_column]]
        .drop_duplicates(subset=["ticker", "prediction_session"])
        .rename(columns={realized_column: "realized_day_return"})
        .reset_index(drop=True)
    )


def _build_raw_daily_returns(labeled_frame: pd.DataFrame) -> pd.DataFrame:
    normalized = _normalize_time_columns(labeled_frame, ["session_date", "timestamp"])
    if normalized.empty or "raw_slot_return" not in normalized.columns:
        return pd.DataFrame(columns=["ticker", "prediction_session", "next_day_realized_return"])
    return (
        normalized.groupby(["ticker", "session_date"], as_index=False)
        .agg(next_day_realized_return=("raw_slot_return", "sum"))
        .rename(columns={"session_date": "prediction_session"})
    )


def _build_book(
    forecasts: pd.DataFrame,
    *,
    score_column: str,
    n_long: int,
    n_short: int,
    book_mode: str,
    tail_fraction: float | None,
) -> pd.DataFrame:
    score_frame = forecasts.copy()
    score_frame["score_q50"] = score_frame[score_column]
    book = PortfolioBuilder().rank_and_size(
        score_frame,
        n_long=n_long,
        n_short=n_short,
        score_column=score_column,
        book_mode=book_mode,
        tail_fraction=tail_fraction,
    )
    return book.frame


def _infer_short_book_size(book_frame: pd.DataFrame) -> int:
    if book_frame.empty:
        return 0
    short_counts = (
        book_frame[book_frame["side"] == "short"]
        .groupby("prediction_session")
        .size()
    )
    if short_counts.empty:
        return 0
    return int(short_counts.median())


def _safe_summary_stat(series: pd.Series, stat: str) -> float:
    clean = series.dropna()
    if clean.empty:
        return float("nan")
    if stat == "mean":
        return float(clean.mean())
    if stat == "median":
        return float(clean.median())
    raise ValueError(f"Unsupported stat: {stat}")


def compute_rank_ic_for_score(
    forecasts: pd.DataFrame,
    score_column: str,
    *,
    realized_column: str = "realized_day_return",
) -> pd.DataFrame:
    merged = forecasts[["ticker", "prediction_session", score_column, realized_column]].copy()
    merged = merged.rename(columns={score_column: "ranking_score", realized_column: "realized_day_return"})
    if merged.empty:
        return pd.DataFrame(columns=["prediction_session", "rank_ic"])

    daily_records: list[dict[str, object]] = []
    for prediction_session, day_frame in merged.groupby("prediction_session", sort=True):
        ranked_scores = day_frame["ranking_score"].rank(method="average")
        ranked_returns = day_frame["realized_day_return"].rank(method="average")
        if ranked_scores.nunique(dropna=True) <= 1 or ranked_returns.nunique(dropna=True) <= 1:
            rank_ic = float("nan")
        else:
            rank_ic = float(ranked_scores.corr(ranked_returns))
        daily_records.append({"prediction_session": prediction_session, "rank_ic": rank_ic})

    daily_ic = pd.DataFrame.from_records(daily_records)
    summary = pd.DataFrame(
        [
            {"metric": "mean_rank_ic", "value": _safe_summary_stat(daily_ic["rank_ic"], "mean")},
            {"metric": "median_rank_ic", "value": _safe_summary_stat(daily_ic["rank_ic"], "median")},
            {"metric": "positive_rank_ic_rate", "value": float((daily_ic["rank_ic"] > 0).mean())},
        ]
    )
    return pd.concat([daily_ic, summary], ignore_index=True, sort=False)


def _extract_metric(metrics_summary: pd.DataFrame, cost_bps: int, column: str = "annualized_sharpe") -> float:
    row = metrics_summary.loc[metrics_summary["cost_bps"] == cost_bps]
    if row.empty:
        return float("nan")
    return float(row.iloc[0][column])


def _build_daily_side_returns(position_returns: pd.DataFrame, n_short: int) -> pd.DataFrame:
    positions = position_returns.copy()
    positions["hit"] = np.where(
        positions["side"] == "long",
        positions["realized_day_return"] > 0,
        positions["realized_day_return"] < 0,
    )

    daily_side = (
        positions.groupby(["prediction_session", "side"], as_index=False)
        .agg(
            basket_return=("realized_day_return", "mean"),
            side_pnl=("gross_position_return", "sum"),
            hit_rate=("hit", "mean"),
        )
        .sort_values(["prediction_session", "side"])
        .reset_index(drop=True)
    )

    long_side = daily_side[daily_side["side"] == "long"].rename(
        columns={"basket_return": "long_basket_return", "side_pnl": "long_side_pnl", "hit_rate": "long_hit_rate_daily"}
    )
    short_side = daily_side[daily_side["side"] == "short"].rename(
        columns={"basket_return": "short_basket_return", "side_pnl": "short_side_pnl", "hit_rate": "short_hit_rate_daily"}
    )
    merged = long_side.merge(
        short_side[["prediction_session", "short_basket_return", "short_side_pnl", "short_hit_rate_daily"]],
        on="prediction_session",
        how="outer",
        validate="one_to_one",
    ).sort_values("prediction_session")
    merged["long_short_spread"] = merged["long_basket_return"] - merged["short_basket_return"]
    merged["gross_long_short_pnl"] = merged["long_side_pnl"] + merged["short_side_pnl"]

    strongest_long = (
        positions[positions["side"] == "long"]
        .sort_values(["prediction_session", "ranking_score", "ticker"], ascending=[True, False, True])
        .groupby("prediction_session", as_index=False)
        .first()[["prediction_session", "realized_day_return"]]
        .rename(columns={"realized_day_return": "top1_long_return"})
    )
    strongest_short = (
        positions[positions["side"] == "short"]
        .sort_values(["prediction_session", "ranking_score", "ticker"], ascending=[True, True, True])
        .groupby("prediction_session", as_index=False)
        .first()[["prediction_session", "realized_day_return"]]
        .rename(columns={"realized_day_return": "top1_short_return"})
    )
    strongest = strongest_long.merge(strongest_short, on="prediction_session", how="outer", validate="one_to_one")
    strongest["top1_capture"] = strongest["top1_long_return"] - strongest["top1_short_return"]

    merged = merged.merge(strongest, on="prediction_session", how="left", validate="one_to_one")
    merged["top5_capture"] = merged["long_short_spread"]
    merged["short_book_size"] = n_short
    return merged.reset_index(drop=True)


def _build_score_summary(
    *,
    forecasts: pd.DataFrame,
    score_column: str,
    n_long: int,
    n_short: int,
    costs_bps: tuple[int, ...],
    realized_column: str = "realized_day_return",
    book_mode: str = "fixed_count",
    tail_fraction: float | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    rank_ic = compute_rank_ic_for_score(forecasts, score_column, realized_column=realized_column)
    realized = _build_realized_returns(forecasts, realized_column=realized_column)
    book_frame = _build_book(
        forecasts,
        score_column=score_column,
        n_long=n_long,
        n_short=n_short,
        book_mode=book_mode,
        tail_fraction=tail_fraction,
    )
    metrics = BacktestRunner().run(book_frame, realized, costs_bps=costs_bps)
    daily_side = _build_daily_side_returns(metrics.position_returns, n_short=max(1, _infer_short_book_size(book_frame)))

    positions = metrics.position_returns.copy()
    long_hit_rate = float((positions.loc[positions["side"] == "long", "realized_day_return"] > 0).mean())
    short_hit_rate = float((positions.loc[positions["side"] == "short", "realized_day_return"] < 0).mean())

    summary_row = pd.DataFrame(
        [
            {
                "score_column": score_column,
                "prediction_days": int(metrics.daily_returns["prediction_session"].nunique()),
                "mean_rank_ic": float(rank_ic.loc[rank_ic["metric"] == "mean_rank_ic", "value"].iloc[0]),
                "median_rank_ic": float(rank_ic.loc[rank_ic["metric"] == "median_rank_ic", "value"].iloc[0]),
                "positive_rank_ic_rate": float(rank_ic.loc[rank_ic["metric"] == "positive_rank_ic_rate", "value"].iloc[0]),
                "gross_sharpe": _extract_metric(metrics.summary, 0),
                "net_sharpe_10bps": _extract_metric(metrics.summary, 10),
                "net_sharpe_20bps": _extract_metric(metrics.summary, 20),
                "net_sharpe_40bps": _extract_metric(metrics.summary, 40),
                "long_hit_rate": long_hit_rate,
                "short_hit_rate": short_hit_rate,
                "top1_capture_mean": float(daily_side["top1_capture"].mean()),
                "top5_capture_mean": float(daily_side["top5_capture"].mean()),
                "mean_daily_long_return": float(daily_side["long_basket_return"].mean()),
                "mean_daily_short_return": float(daily_side["short_basket_return"].mean()),
                "mean_daily_long_short_spread": float(daily_side["long_short_spread"].mean()),
                "book_mode": book_mode,
                "tail_fraction": tail_fraction if book_mode == "tail_fraction" else float("nan"),
            }
        ]
    )
    return summary_row, daily_side, book_frame, rank_ic


def _build_score_relationships(forecasts: pd.DataFrame) -> pd.DataFrame:
    score_columns = [column for column in ["score_plain", "score_z", "score_risk"] if column in forecasts.columns]
    records: list[dict[str, object]] = []
    for index, left in enumerate(score_columns):
        for right in score_columns[index + 1 :]:
            pair = forecasts[[left, right]].dropna()
            if pair.empty:
                pearson = float("nan")
                spearman = float("nan")
                same_sign_rate = float("nan")
            else:
                pearson = float(pair[left].corr(pair[right], method="pearson"))
                spearman = float(pair[left].corr(pair[right], method="spearman"))
                same_sign_rate = float((np.sign(pair[left]) == np.sign(pair[right])).mean())
            records.append(
                {
                    "left_score": left,
                    "right_score": right,
                    "pearson_correlation": pearson,
                    "spearman_correlation": spearman,
                    "same_sign_rate": same_sign_rate,
                }
            )
    return pd.DataFrame.from_records(records)


def _build_beta_summary(forecasts: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    beta_frame = forecasts[["ticker", "basket_id", "beta_used"]].copy()
    beta_frame["clamp_hit_zero"] = np.isclose(beta_frame["beta_used"], 0.0)
    beta_frame["clamp_hit_three"] = np.isclose(beta_frame["beta_used"], 3.0)
    beta_frame["clamp_hit_any"] = beta_frame["clamp_hit_zero"] | beta_frame["clamp_hit_three"]

    overall = pd.DataFrame(
        [
            {
                "mean_beta": float(beta_frame["beta_used"].mean()),
                "median_beta": float(beta_frame["beta_used"].median()),
                "min_beta": float(beta_frame["beta_used"].min()),
                "max_beta": float(beta_frame["beta_used"].max()),
                "clamp_hit_zero_count": int(beta_frame["clamp_hit_zero"].sum()),
                "clamp_hit_three_count": int(beta_frame["clamp_hit_three"].sum()),
                "clamp_hit_any_count": int(beta_frame["clamp_hit_any"].sum()),
                "unique_tickers_with_clamps": int(beta_frame.loc[beta_frame["clamp_hit_any"], "ticker"].nunique()),
            }
        ]
    )
    by_ticker = (
        beta_frame.groupby(["ticker", "basket_id"], as_index=False)
        .agg(
            mean_beta=("beta_used", "mean"),
            median_beta=("beta_used", "median"),
            min_beta=("beta_used", "min"),
            max_beta=("beta_used", "max"),
            clamp_hit_zero_count=("clamp_hit_zero", "sum"),
            clamp_hit_three_count=("clamp_hit_three", "sum"),
            clamp_hit_any_count=("clamp_hit_any", "sum"),
            observation_count=("beta_used", "size"),
        )
        .sort_values(["clamp_hit_any_count", "ticker"], ascending=[False, True])
        .reset_index(drop=True)
    )
    return overall, by_ticker


def _normalize_data_quality_events(events: pd.DataFrame | None) -> pd.DataFrame:
    if events is None or events.empty:
        return pd.DataFrame(columns=EXPECTED_DATA_QUALITY_COLUMNS)
    normalized = events.copy()
    for column in EXPECTED_DATA_QUALITY_COLUMNS:
        if column not in normalized.columns:
            normalized[column] = np.nan
    normalized = _normalize_time_columns(normalized, ["prediction_session"])
    return normalized[EXPECTED_DATA_QUALITY_COLUMNS].copy()


def _build_data_quality_reports(
    forecasts: pd.DataFrame,
    data_quality_events: pd.DataFrame | None,
    data_quality_summary: pd.DataFrame | None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    events = _normalize_data_quality_events(data_quality_events)
    summary = data_quality_summary.copy() if data_quality_summary is not None else pd.DataFrame()
    prediction_days = int(forecasts["prediction_session"].nunique())
    baskets = sorted(forecasts["basket_id"].dropna().unique().tolist()) if "basket_id" in forecasts.columns else []

    overview_records = [
        {"metric": "prediction_days", "value": float(prediction_days)},
        {
            "metric": "basket_dates_dropped",
            "value": float(
                summary.loc[summary["metric"] == "basket_dates_dropped", "value"].iloc[0]
                if not summary.empty and (summary["metric"] == "basket_dates_dropped").any()
                else 0.0
            ),
        },
        {
            "metric": "single_slot_imputations_used",
            "value": float(
                summary.loc[summary["metric"] == "single_slot_imputations_used", "value"].iloc[0]
                if not summary.empty and (summary["metric"] == "single_slot_imputations_used").any()
                else 0.0
            ),
        },
    ]
    overview = pd.DataFrame.from_records(overview_records)

    by_basket_records: list[dict[str, object]] = []
    for basket_id in baskets:
        basket_events = events[events["basket_id"] == basket_id]
        by_basket_records.append(
            {
                "basket_id": basket_id,
                "basket_dates_dropped": int(
                    basket_events.loc[basket_events["event_type"] == "drop", ["basket_id", "prediction_session"]]
                    .drop_duplicates()
                    .shape[0]
                ),
                "single_slot_imputations_used": int((basket_events["event_type"] == "single_slot_imputation").sum()),
            }
        )
    by_basket = pd.DataFrame.from_records(by_basket_records)

    worst_tickers = (
        events[events["event_type"] == "drop"]
        .groupby("ticker", as_index=False)
        .size()
        .rename(columns={"size": "drop_count"})
        .sort_values(["drop_count", "ticker"], ascending=[False, True])
        .reset_index(drop=True)
    )
    return overview, by_basket, worst_tickers


def _build_beta_effect_comparison(
    forecasts: pd.DataFrame,
    labeled_frame: pd.DataFrame | None,
    *,
    benchmark_ticker: str,
    n_long: int,
    n_short: int,
    costs_bps: tuple[int, ...],
) -> pd.DataFrame:
    if labeled_frame is None or labeled_frame.empty:
        return pd.DataFrame()

    labeled = _normalize_time_columns(labeled_frame, ["session_date", "timestamp"])
    if "raw_slot_return" not in labeled.columns:
        return pd.DataFrame()

    raw_daily = (
        labeled.groupby(["ticker", "session_date"], as_index=False)
        .agg(raw_day_return=("raw_slot_return", "sum"))
        .rename(columns={"session_date": "prediction_session"})
    )
    spy_daily = raw_daily[raw_daily["ticker"] == benchmark_ticker][["prediction_session", "raw_day_return"]].rename(
        columns={"raw_day_return": "spy_raw_day_return"}
    )
    candidate_daily = raw_daily[raw_daily["ticker"] != benchmark_ticker].merge(
        spy_daily,
        on="prediction_session",
        how="left",
        validate="many_to_one",
    )
    candidate_daily["realized_unit_beta_day_return"] = candidate_daily["raw_day_return"] - candidate_daily["spy_raw_day_return"]

    unit_beta = forecasts.copy()
    unit_beta["score_plain_unit_beta"] = unit_beta["stock_q50_path"].apply(sum) - unit_beta["spy_q50_path"].apply(sum)
    unit_beta = unit_beta.merge(
        candidate_daily[["ticker", "prediction_session", "realized_unit_beta_day_return"]],
        on=["ticker", "prediction_session"],
        how="left",
        validate="many_to_one",
    )

    adjusted_summary, _, _, _ = _build_score_summary(
        forecasts=forecasts,
        score_column="score_plain",
        n_long=n_long,
        n_short=n_short,
        costs_bps=costs_bps,
        realized_column="realized_day_return",
    )
    unit_summary, _, _, _ = _build_score_summary(
        forecasts=unit_beta,
        score_column="score_plain_unit_beta",
        n_long=n_long,
        n_short=n_short,
        costs_bps=costs_bps,
        realized_column="realized_unit_beta_day_return",
    )
    adjusted_summary["variant"] = "beta_adjusted"
    unit_summary["variant"] = "unit_beta"
    return pd.concat([adjusted_summary, unit_summary], ignore_index=True, sort=False)


def _build_score_behavior_summary(
    forecasts: pd.DataFrame,
    *,
    score_columns: tuple[str, ...],
    realized_column: str = "realized_day_return",
) -> pd.DataFrame:
    records: list[dict[str, object]] = []
    for score_column in score_columns:
        if score_column not in forecasts.columns:
            continue
        frame = forecasts[[score_column, realized_column]].dropna().copy()
        if frame.empty:
            continue
        predicted_sign = np.sign(frame[score_column])
        realized_sign = np.sign(frame[realized_column])
        nonzero = (predicted_sign != 0) & (realized_sign != 0)
        sign_accuracy = float((predicted_sign[nonzero] == realized_sign[nonzero]).mean()) if nonzero.any() else float("nan")
        signed_realized = predicted_sign * frame[realized_column]
        positive_predictions = frame[score_column] > 0
        negative_predictions = frame[score_column] < 0
        records.append(
            {
                "score_column": score_column,
                "observation_count": int(len(frame)),
                "sign_accuracy": sign_accuracy,
                "positive_prediction_rate": float(positive_predictions.mean()),
                "negative_prediction_rate": float(negative_predictions.mean()),
                "long_hit_rate": float((frame.loc[positive_predictions, realized_column] > 0).mean()) if positive_predictions.any() else float("nan"),
                "short_hit_rate": float((frame.loc[negative_predictions, realized_column] < 0).mean()) if negative_predictions.any() else float("nan"),
                "mean_abs_score": float(frame[score_column].abs().mean()),
                "pearson_to_realized": float(frame[score_column].corr(frame[realized_column], method="pearson")),
                "spearman_to_realized": float(frame[score_column].corr(frame[realized_column], method="spearman")),
                "mean_signed_realized_return": float(signed_realized.mean()),
            }
        )
    return pd.DataFrame.from_records(records)


def _build_signal_strength_summary(
    forecasts: pd.DataFrame,
    *,
    score_column: str,
    realized_column: str = "realized_day_return",
) -> pd.DataFrame:
    frame = forecasts[["prediction_session", score_column, realized_column]].dropna().copy()
    if frame.empty:
        return pd.DataFrame()
    frame["abs_score"] = frame[score_column].abs()
    frame["strength_bucket"] = "weak"
    for prediction_session, day_frame in frame.groupby("prediction_session", sort=True):
        median_abs_score = float(day_frame["abs_score"].median())
        mask = frame["prediction_session"] == prediction_session
        frame.loc[mask & (frame["abs_score"] >= median_abs_score), "strength_bucket"] = "strong"

    records: list[dict[str, object]] = []
    for strength_bucket, bucket_frame in frame.groupby("strength_bucket", sort=True):
        predicted_sign = np.sign(bucket_frame[score_column])
        realized_sign = np.sign(bucket_frame[realized_column])
        nonzero = (predicted_sign != 0) & (realized_sign != 0)
        sign_accuracy = (
            float((predicted_sign[nonzero] == realized_sign[nonzero]).mean())
            if nonzero.any()
            else float("nan")
        )
        records.append(
            {
                "strength_bucket": strength_bucket,
                "observation_count": int(len(bucket_frame)),
                "mean_abs_score": float(bucket_frame["abs_score"].mean()),
                "sign_accuracy": sign_accuracy,
                "mean_signed_realized_return": float((predicted_sign * bucket_frame[realized_column]).mean()),
                "mean_realized_return": float(bucket_frame[realized_column].mean()),
            }
        )
    return pd.DataFrame.from_records(records).sort_values("strength_bucket").reset_index(drop=True)


def _build_fractional_tail_book(
    forecasts: pd.DataFrame,
    *,
    score_column: str,
    tail_fraction: float,
) -> pd.DataFrame:
    if tail_fraction <= 0 or tail_fraction >= 0.5:
        raise ValueError("tail_fraction must be between 0 and 0.5.")

    extra_columns = [
        column
        for column in ["basket_id", "score_plain", "score_risk", "score_z", "score_q50"]
        if column in forecasts.columns
    ]
    records: list[dict[str, object]] = []
    for signal_session, day_frame in forecasts.groupby("signal_session", sort=True):
        ordered = day_frame.sort_values([score_column, "ticker"], ascending=[False, True]).reset_index(drop=True)
        n_each = max(1, int(np.ceil(len(ordered) * tail_fraction)))
        longs = ordered.head(n_each).copy()
        shorts = ordered.tail(n_each).copy()
        long_weight = 0.5 / n_each
        short_weight = -0.5 / n_each

        for rank, (_, row) in enumerate(longs.iterrows(), start=1):
            record = {
                "signal_session": signal_session,
                "prediction_session": row["prediction_session"],
                "ticker": row["ticker"],
                "side": "long",
                "rank": rank,
                "weight": long_weight,
                "ranking_score_column": score_column,
                "ranking_score": row[score_column],
                "tail_fraction": tail_fraction,
                "tail_size": n_each,
            }
            for column in extra_columns:
                record[column] = row[column]
            records.append(record)
        for rank, (_, row) in enumerate(shorts.iterrows(), start=1):
            record = {
                "signal_session": signal_session,
                "prediction_session": row["prediction_session"],
                "ticker": row["ticker"],
                "side": "short",
                "rank": rank,
                "weight": short_weight,
                "ranking_score_column": score_column,
                "ranking_score": row[score_column],
                "tail_fraction": tail_fraction,
                "tail_size": n_each,
            }
            for column in extra_columns:
                record[column] = row[column]
            records.append(record)
    return pd.DataFrame.from_records(records).sort_values(["signal_session", "side", "ticker"]).reset_index(drop=True)


def _build_tail_filter_comparison(
    forecasts: pd.DataFrame,
    *,
    score_column: str,
    tail_fractions: tuple[float, ...],
    n_short: int,
    costs_bps: tuple[int, ...],
    realized_column: str = "realized_day_return",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    realized = _build_realized_returns(forecasts, realized_column=realized_column)
    summary_frames: list[pd.DataFrame] = []
    daily_frames: list[pd.DataFrame] = []

    for tail_fraction in tail_fractions:
        book_frame = _build_fractional_tail_book(forecasts, score_column=score_column, tail_fraction=tail_fraction)
        metrics = BacktestRunner().run(book_frame, realized, costs_bps=costs_bps)
        daily_side = _build_daily_side_returns(metrics.position_returns, n_short=int(book_frame["tail_size"].max()))
        positions = metrics.position_returns.copy()
        summary_frames.append(
            pd.DataFrame(
                [
                    {
                        "score_column": score_column,
                        "tail_fraction": tail_fraction,
                        "tail_size": int(book_frame["tail_size"].median()),
                        "prediction_days": int(metrics.daily_returns["prediction_session"].nunique()),
                        "gross_sharpe": _extract_metric(metrics.summary, 0),
                        "net_sharpe_10bps": _extract_metric(metrics.summary, 10),
                        "net_sharpe_20bps": _extract_metric(metrics.summary, 20),
                        "net_sharpe_40bps": _extract_metric(metrics.summary, 40),
                        "long_hit_rate": float((positions.loc[positions["side"] == "long", "realized_day_return"] > 0).mean()),
                        "short_hit_rate": float((positions.loc[positions["side"] == "short", "realized_day_return"] < 0).mean()),
                        "top1_capture_mean": float(daily_side["top1_capture"].mean()),
                        "topN_capture_mean": float(daily_side["top5_capture"].mean()),
                        "mean_daily_long_short_spread": float(daily_side["long_short_spread"].mean()),
                    }
                ]
            )
        )
        daily_frames.append(daily_side.assign(score_column=score_column, tail_fraction=tail_fraction))

    return (
        pd.concat(summary_frames, ignore_index=True, sort=False),
        pd.concat(daily_frames, ignore_index=True, sort=False),
    )


def _build_execution_sensitivity(
    forecasts: pd.DataFrame,
    *,
    book_frame: pd.DataFrame,
    costs_bps: tuple[int, ...],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if "realized_alpha_path" not in forecasts.columns:
        return pd.DataFrame(), pd.DataFrame()

    execution_modes = {
        "exact_open": lambda path: float(np.sum(path)),
        "delayed_open_1h": lambda path: float(np.sum(path[1:])) if len(path) > 1 else float("nan"),
        "first_15m_avg_proxy": lambda path: float((0.875 * path[0]) + np.sum(path[1:])) if len(path) > 0 else float("nan"),
    }
    summary_frames: list[pd.DataFrame] = []
    daily_frames: list[pd.DataFrame] = []

    for mode_name, reducer in execution_modes.items():
        realized = forecasts[["ticker", "prediction_session", "realized_alpha_path"]].copy()
        realized["realized_day_return"] = realized["realized_alpha_path"].apply(lambda path: reducer(list(map(float, path))))
        realized = realized[["ticker", "prediction_session", "realized_day_return"]].drop_duplicates(
            subset=["ticker", "prediction_session"]
        )
        metrics = BacktestRunner().run(book_frame, realized, costs_bps=costs_bps)
        daily_side = _build_daily_side_returns(metrics.position_returns, n_short=int((book_frame["side"] == "short").sum() / book_frame["prediction_session"].nunique()))
        positions = metrics.position_returns.copy()
        summary_frames.append(
            pd.DataFrame(
                [
                    {
                        "execution_mode": mode_name,
                        "prediction_days": int(metrics.daily_returns["prediction_session"].nunique()),
                        "gross_sharpe": _extract_metric(metrics.summary, 0),
                        "net_sharpe_10bps": _extract_metric(metrics.summary, 10),
                        "net_sharpe_20bps": _extract_metric(metrics.summary, 20),
                        "net_sharpe_40bps": _extract_metric(metrics.summary, 40),
                        "long_hit_rate": float((positions.loc[positions["side"] == "long", "realized_day_return"] > 0).mean()),
                        "short_hit_rate": float((positions.loc[positions["side"] == "short", "realized_day_return"] < 0).mean()),
                        "mean_daily_long_short_spread": float(daily_side["long_short_spread"].mean()),
                    }
                ]
            )
        )
        daily_frames.append(daily_side.assign(execution_mode=mode_name))

    return (
        pd.concat(summary_frames, ignore_index=True, sort=False),
        pd.concat(daily_frames, ignore_index=True, sort=False),
    )


def _empty_slice_summary(slice_name_column: str, slice_name: str) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                slice_name_column: slice_name,
                "prediction_days": 0,
                "gross_sharpe": float("nan"),
                "net_sharpe_10bps": float("nan"),
                "net_sharpe_20bps": float("nan"),
                "net_sharpe_40bps": float("nan"),
                "mean_daily_return": float("nan"),
                "long_hit_rate": float("nan"),
                "short_hit_rate": float("nan"),
                "mean_daily_long_short_spread": float("nan"),
            }
        ]
    )


def _summarize_book_slice(
    *,
    book_frame: pd.DataFrame,
    realized: pd.DataFrame,
    costs_bps: tuple[int, ...],
    slice_name_column: str,
    slice_name: str,
) -> pd.DataFrame:
    if book_frame.empty:
        return _empty_slice_summary(slice_name_column, slice_name)

    metrics = BacktestRunner().run(book_frame, realized, costs_bps=costs_bps)
    positions = metrics.position_returns.copy()
    daily_side = _build_daily_side_returns(positions, n_short=max(1, _infer_short_book_size(book_frame)))
    long_positions = positions[positions["side"] == "long"]
    short_positions = positions[positions["side"] == "short"]
    return pd.DataFrame(
        [
            {
                slice_name_column: slice_name,
                "prediction_days": int(metrics.daily_returns["prediction_session"].nunique()),
                "gross_sharpe": _extract_metric(metrics.summary, 0),
                "net_sharpe_10bps": _extract_metric(metrics.summary, 10),
                "net_sharpe_20bps": _extract_metric(metrics.summary, 20),
                "net_sharpe_40bps": _extract_metric(metrics.summary, 40),
                "mean_daily_return": float(metrics.daily_returns["gross_return"].mean()),
                "long_hit_rate": float((long_positions["realized_day_return"] > 0).mean()) if not long_positions.empty else float("nan"),
                "short_hit_rate": float((short_positions["realized_day_return"] < 0).mean()) if not short_positions.empty else float("nan"),
                "mean_daily_long_short_spread": float(daily_side["long_short_spread"].mean()) if not daily_side.empty else float("nan"),
            }
        ]
    )


def _build_basket_book_comparison(
    *,
    book_frame: pd.DataFrame,
    realized: pd.DataFrame,
    costs_bps: tuple[int, ...],
) -> pd.DataFrame:
    if "basket_id" not in book_frame.columns:
        return pd.DataFrame()

    slices = {
        "combined": book_frame,
        "growth_only": book_frame[book_frame["basket_id"] == "growth"].copy(),
        "defensive_only": book_frame[book_frame["basket_id"] == "defensive_value"].copy(),
    }
    frames = [
        _summarize_book_slice(
            book_frame=slice_frame,
            realized=realized,
            costs_bps=costs_bps,
            slice_name_column="basket_slice",
            slice_name=slice_name,
        )
        for slice_name, slice_frame in slices.items()
    ]
    return pd.concat(frames, ignore_index=True, sort=False)


def _build_side_book_comparison(
    *,
    book_frame: pd.DataFrame,
    realized: pd.DataFrame,
    costs_bps: tuple[int, ...],
) -> pd.DataFrame:
    slices = {
        "long_short": book_frame,
        "long_only": book_frame[book_frame["side"] == "long"].copy(),
        "short_only": book_frame[book_frame["side"] == "short"].copy(),
    }
    frames = [
        _summarize_book_slice(
            book_frame=slice_frame,
            realized=realized,
            costs_bps=costs_bps,
            slice_name_column="side_slice",
            slice_name=slice_name,
        )
        for slice_name, slice_frame in slices.items()
    ]
    return pd.concat(frames, ignore_index=True, sort=False)


def _build_basket_side_summary(position_returns: pd.DataFrame) -> pd.DataFrame:
    if position_returns.empty or "basket_id" not in position_returns.columns:
        return pd.DataFrame()

    positions = position_returns.copy()
    positions["hit"] = np.where(
        positions["side"] == "long",
        positions["realized_day_return"] > 0,
        positions["realized_day_return"] < 0,
    )
    total_abs_pnl = float(positions["gross_position_return"].abs().sum())
    summary = (
        positions.groupby(["basket_id", "side"], as_index=False)
        .agg(
            mean_realized_return=("realized_day_return", "mean"),
            mean_gross_pnl=("gross_position_return", "mean"),
            total_gross_pnl=("gross_position_return", "sum"),
            hit_rate=("hit", "mean"),
            position_count=("ticker", "size"),
        )
        .sort_values(["basket_id", "side"])
        .reset_index(drop=True)
    )
    summary["contribution_share"] = summary["total_gross_pnl"].abs() / total_abs_pnl if total_abs_pnl > 0 else 0.0
    return summary


def _normalize_execution_bars(
    raw_execution_bars: pd.DataFrame,
    *,
    market_timezone: str = "America/New_York",
) -> pd.DataFrame:
    if raw_execution_bars.empty:
        return pd.DataFrame(columns=["ticker", "timestamp", "prediction_session", "clock_time", "open", "high", "low", "close", "volume"])

    frame = raw_execution_bars[["ticker", "timestamp", "open", "high", "low", "close", "volume"]].copy()
    timestamps = pd.to_datetime(frame["timestamp"])
    if timestamps.dt.tz is None:
        timestamps = timestamps.dt.tz_localize(ZoneInfo(market_timezone))
    else:
        timestamps = timestamps.dt.tz_convert(ZoneInfo(market_timezone))
    frame["timestamp"] = timestamps
    frame["clock_time"] = frame["timestamp"].dt.strftime("%H:%M")
    frame = frame[(frame["clock_time"] >= "09:30") & (frame["clock_time"] <= "15:55")].copy()
    frame["prediction_session"] = frame["timestamp"].dt.normalize().dt.tz_convert("UTC")
    return frame.sort_values(["ticker", "timestamp"]).reset_index(drop=True)


def _bar_ohlc4(row: pd.Series) -> float:
    return float((row["open"] + row["high"] + row["low"] + row["close"]) / 4.0)


def _weighted_average(values: list[float], weights: list[float]) -> float:
    weight_sum = float(np.sum(weights))
    if weight_sum <= 0:
        return float(np.mean(values))
    return float(np.average(values, weights=np.array(weights, dtype=float)))


def _build_execution_price_frame(raw_execution_bars: pd.DataFrame) -> pd.DataFrame:
    normalized = _normalize_execution_bars(raw_execution_bars)
    required_times = ("09:30", "09:35", "09:40", "15:55")
    records: list[dict[str, object]] = []

    for (ticker, prediction_session), group in normalized.groupby(["ticker", "prediction_session"], sort=True):
        ordered = group.sort_values("timestamp").reset_index(drop=True)
        by_time = {
            clock_time: time_frame.iloc[0]
            for clock_time, time_frame in ordered.groupby("clock_time", sort=True)
        }
        missing_times: list[str] = []
        for clock_time in required_times:
            if clock_time not in by_time:
                missing_times.append(clock_time)
                continue
            row = by_time[clock_time]
            if pd.isna(row["open"]) or pd.isna(row["high"]) or pd.isna(row["low"]) or pd.isna(row["close"]) or pd.isna(row["volume"]):
                missing_times.append(clock_time)

        if missing_times:
            records.append(
                {
                    "ticker": ticker,
                    "prediction_session": prediction_session,
                    "has_required_bars": False,
                    "missing_clock_times": ",".join(missing_times),
                }
            )
            continue

        first_bar = by_time["09:30"]
        first_three = [by_time["09:30"], by_time["09:35"], by_time["09:40"]]
        exit_close = float(by_time["15:55"]["close"])
        exact_open_entry = float(first_bar["open"])
        first_5m_avg_entry = _bar_ohlc4(first_bar)
        first_15m_avg_entry = _weighted_average(
            [_bar_ohlc4(row) for row in first_three],
            [float(row["volume"]) for row in first_three],
        )
        records.append(
            {
                "ticker": ticker,
                "prediction_session": prediction_session,
                "has_required_bars": True,
                "missing_clock_times": "",
                "exact_open_return": float(np.log(exit_close / exact_open_entry)),
                "first_5m_avg_return": float(np.log(exit_close / first_5m_avg_entry)),
                "first_15m_avg_return": float(np.log(exit_close / first_15m_avg_entry)),
            }
        )

    return pd.DataFrame.from_records(records)


def _build_execution_overlay_realized(
    *,
    valid_book: pd.DataFrame,
    execution_prices: pd.DataFrame,
    benchmark_ticker: str,
    mode_column: str,
) -> pd.DataFrame:
    asset_prices = execution_prices[execution_prices["ticker"] != benchmark_ticker][["ticker", "prediction_session", mode_column]].copy()
    spy_prices = execution_prices[execution_prices["ticker"] == benchmark_ticker][["prediction_session", mode_column]].rename(
        columns={mode_column: "spy_return"}
    )
    realized = asset_prices.merge(
        spy_prices,
        on="prediction_session",
        how="left",
        validate="many_to_one",
    )
    realized["realized_day_return"] = realized[mode_column] - realized["spy_return"]
    realized = realized[["ticker", "prediction_session", "realized_day_return"]]
    return realized[realized["prediction_session"].isin(valid_book["prediction_session"].unique())].reset_index(drop=True)


def generate_near_open_execution_analysis(
    *,
    book_frame: pd.DataFrame,
    raw_execution_bars: pd.DataFrame,
    benchmark_ticker: str = "SPY",
    costs_bps: tuple[int, ...] = (10, 20, 40),
    execution_mode: str = "strict",
) -> dict[str, pd.DataFrame]:
    if execution_mode != STRICT_EXECUTION_MODE:
        raise ValueError(f"Unsupported execution_mode: {execution_mode!r}")

    normalized_book = _normalize_time_columns(book_frame, ["signal_session", "prediction_session"])
    execution_prices = _build_execution_price_frame(raw_execution_bars)
    prediction_sessions = sorted(pd.to_datetime(normalized_book["prediction_session"], utc=True).unique().tolist()) if not normalized_book.empty else []
    price_lookup = {
        (row.ticker, row.prediction_session): row
        for row in execution_prices.itertuples(index=False)
    }
    quality_records: list[dict[str, object]] = []
    valid_sessions: list[pd.Timestamp] = []

    for prediction_session, day_book in normalized_book.groupby("prediction_session", sort=True):
        day_valid = True
        required_tickers = sorted(set(day_book["ticker"].tolist()) | {benchmark_ticker})
        for ticker in required_tickers:
            row = price_lookup.get((ticker, prediction_session))
            if row is None:
                quality_records.append(
                    {
                        "prediction_session": prediction_session,
                        "ticker": ticker,
                        "execution_mode": execution_mode,
                        "execution_mode_label": STRICT_EXECUTION_MODE_LABEL,
                        "event_type": "drop",
                        "reason": "missing_execution_bars",
                        "action_taken": "drop_prediction_day",
                    }
                )
                day_valid = False
                continue
            if not bool(row.has_required_bars):
                quality_records.append(
                    {
                        "prediction_session": prediction_session,
                        "ticker": ticker,
                        "execution_mode": execution_mode,
                        "execution_mode_label": STRICT_EXECUTION_MODE_LABEL,
                        "event_type": "drop",
                        "reason": f"missing_required_bars:{row.missing_clock_times}",
                        "action_taken": "drop_prediction_day",
                    }
                )
                day_valid = False
        if day_valid:
            valid_sessions.append(prediction_session)

    valid_book = normalized_book[normalized_book["prediction_session"].isin(valid_sessions)].copy()
    execution_quality = pd.DataFrame.from_records(
        quality_records,
        columns=[
            "prediction_session",
            "ticker",
            "execution_mode",
            "execution_mode_label",
            "event_type",
            "reason",
            "action_taken",
        ],
    )
    mode_columns = {
        "exact_open": "exact_open_return",
        "first_5m_avg": "first_5m_avg_return",
        "first_15m_avg": "first_15m_avg_return",
    }
    summary_frames: list[pd.DataFrame] = []
    daily_frames: list[pd.DataFrame] = []
    position_frames: list[pd.DataFrame] = []
    return_frames: list[pd.DataFrame] = []
    total_prediction_days = len(prediction_sessions)

    for mode_name, mode_column in mode_columns.items():
        realized = _build_execution_overlay_realized(
            valid_book=valid_book,
            execution_prices=execution_prices,
            benchmark_ticker=benchmark_ticker,
            mode_column=mode_column,
        )
        metrics = BacktestRunner().run(valid_book, realized, costs_bps=costs_bps)
        daily_side = _build_daily_side_returns(metrics.position_returns, n_short=max(1, _infer_short_book_size(valid_book)))
        positions = metrics.position_returns.copy()
        summary_frames.append(
            pd.DataFrame(
                [
                    {
                        "execution_mode": mode_name,
                        "execution_mode_label": STRICT_EXECUTION_MODE_LABEL,
                        "strict_execution_mode": execution_mode,
                        "entry_formula": EXECUTION_ENTRY_FORMULAS[mode_name],
                        "exit_formula": EXECUTION_EXIT_FORMULA,
                        "alpha_formula": EXECUTION_ALPHA_FORMULA,
                        "prediction_days": int(metrics.daily_returns["prediction_session"].nunique()),
                        "dropped_days": int(total_prediction_days - metrics.daily_returns["prediction_session"].nunique()),
                        "gross_sharpe": _extract_metric(metrics.summary, 0),
                        "net_sharpe_10bps": _extract_metric(metrics.summary, 10),
                        "net_sharpe_20bps": _extract_metric(metrics.summary, 20),
                        "net_sharpe_40bps": _extract_metric(metrics.summary, 40),
                        "long_hit_rate": float((positions.loc[positions["side"] == "long", "realized_day_return"] > 0).mean()) if not positions.empty else float("nan"),
                        "short_hit_rate": float((positions.loc[positions["side"] == "short", "realized_day_return"] < 0).mean()) if not positions.empty else float("nan"),
                        "mean_daily_long_short_spread": float(daily_side["long_short_spread"].mean()) if not daily_side.empty else float("nan"),
                    }
                ]
            )
        )
        position_frames.append(
            metrics.position_returns.assign(
                execution_mode=mode_name,
                execution_mode_label=STRICT_EXECUTION_MODE_LABEL,
                strict_execution_mode=execution_mode,
                entry_formula=EXECUTION_ENTRY_FORMULAS[mode_name],
                exit_formula=EXECUTION_EXIT_FORMULA,
                alpha_formula=EXECUTION_ALPHA_FORMULA,
            )
        )
        return_frames.append(
            metrics.daily_returns.assign(
                execution_mode=mode_name,
                execution_mode_label=STRICT_EXECUTION_MODE_LABEL,
                strict_execution_mode=execution_mode,
                entry_formula=EXECUTION_ENTRY_FORMULAS[mode_name],
                exit_formula=EXECUTION_EXIT_FORMULA,
                alpha_formula=EXECUTION_ALPHA_FORMULA,
            )
        )
        daily_frames.append(
            daily_side.assign(
                execution_mode=mode_name,
                execution_mode_label=STRICT_EXECUTION_MODE_LABEL,
                strict_execution_mode=execution_mode,
                entry_formula=EXECUTION_ENTRY_FORMULAS[mode_name],
                exit_formula=EXECUTION_EXIT_FORMULA,
                alpha_formula=EXECUTION_ALPHA_FORMULA,
            )
        )

    return {
        "execution_sensitivity_comparison": (
            pd.concat(summary_frames, ignore_index=True, sort=False)
            if summary_frames
            else pd.DataFrame(
                columns=[
                    "execution_mode",
                    "execution_mode_label",
                    "strict_execution_mode",
                    "entry_formula",
                    "exit_formula",
                    "alpha_formula",
                    "prediction_days",
                    "dropped_days",
                    "gross_sharpe",
                    "net_sharpe_10bps",
                    "net_sharpe_20bps",
                    "net_sharpe_40bps",
                    "long_hit_rate",
                    "short_hit_rate",
                    "mean_daily_long_short_spread",
                ]
            )
        ),
        "execution_sensitivity_daily": (
            pd.concat(daily_frames, ignore_index=True, sort=False)
            if daily_frames
            else pd.DataFrame()
        ),
        "execution_position_returns": (
            pd.concat(position_frames, ignore_index=True, sort=False)
            if position_frames
            else pd.DataFrame()
        ),
        "execution_daily_returns": (
            pd.concat(return_frames, ignore_index=True, sort=False)
            if return_frames
            else pd.DataFrame()
        ),
        "execution_data_quality": execution_quality,
    }


def summarize_execution_daily_returns(
    execution_daily_returns: pd.DataFrame,
    *,
    costs_bps: tuple[int, ...] = (10, 20, 40),
) -> pd.DataFrame:
    if execution_daily_returns.empty:
        return pd.DataFrame(
            columns=[
                "execution_mode",
                "execution_mode_label",
                "strict_execution_mode",
                "entry_formula",
                "exit_formula",
                "alpha_formula",
                "prediction_days",
                "gross_sharpe",
                "net_sharpe_10bps",
                "net_sharpe_20bps",
                "net_sharpe_40bps",
                "mean_daily_gross_return",
                "mean_daily_long_short_spread",
            ]
        )

    normalized = _normalize_time_columns(execution_daily_returns, ["prediction_session"])
    summary_records: list[dict[str, object]] = []
    for execution_mode, mode_frame in normalized.groupby("execution_mode", sort=True):
        record: dict[str, object] = {
            "execution_mode": execution_mode,
            "execution_mode_label": str(mode_frame["execution_mode_label"].iloc[0]) if "execution_mode_label" in mode_frame.columns else "",
            "strict_execution_mode": str(mode_frame["strict_execution_mode"].iloc[0]) if "strict_execution_mode" in mode_frame.columns else "",
            "entry_formula": str(mode_frame["entry_formula"].iloc[0]) if "entry_formula" in mode_frame.columns else "",
            "exit_formula": str(mode_frame["exit_formula"].iloc[0]) if "exit_formula" in mode_frame.columns else "",
            "alpha_formula": str(mode_frame["alpha_formula"].iloc[0]) if "alpha_formula" in mode_frame.columns else "",
            "prediction_days": int(mode_frame["prediction_session"].nunique()),
            "gross_sharpe": float(_annualized_sharpe(mode_frame["gross_return"])),
            "mean_daily_gross_return": float(mode_frame["gross_return"].mean()),
            # The live execution book is constructed as a 50/50 dollar-neutral portfolio,
            # so gross portfolio return is half of the corresponding long-short spread.
            "mean_daily_long_short_spread": float(2.0 * mode_frame["gross_return"].mean()),
        }
        for cost_bps in costs_bps:
            net_column = f"net_return_{cost_bps}bps"
            if net_column in mode_frame.columns:
                record[f"net_sharpe_{cost_bps}bps"] = float(_annualized_sharpe(mode_frame[net_column]))
            else:
                record[f"net_sharpe_{cost_bps}bps"] = float("nan")
        summary_records.append(record)
    return pd.DataFrame.from_records(summary_records)


def generate_live_reconciliation_outputs(
    *,
    forecasts: pd.DataFrame,
    book_frame: pd.DataFrame,
    labeled_frame: pd.DataFrame,
    raw_execution_bars: pd.DataFrame,
    benchmark_ticker: str = "SPY",
    costs_bps: tuple[int, ...] = (10, 20, 40),
    execution_mode: str = "strict",
) -> dict[str, pd.DataFrame]:
    realized_forecasts = attach_realized_outcomes_to_forecasts(
        forecasts,
        labeled_frame,
        benchmark_ticker=benchmark_ticker,
    )
    normalized_book = _normalize_time_columns(book_frame, ["signal_session", "prediction_session"])
    if "ranking_score_column" not in normalized_book.columns:
        normalized_book["ranking_score_column"] = "score_q50" if "score_q50" in normalized_book.columns else "score_z"
    if "ranking_score" not in normalized_book.columns:
        normalized_book["ranking_score"] = normalized_book.apply(
            lambda row: row[row["ranking_score_column"]]
            if row["ranking_score_column"] in normalized_book.columns
            else row.get("score_q50", np.nan),
            axis=1,
        )
    if "book_mode" not in normalized_book.columns:
        normalized_book["book_mode"] = "fixed_count"
    if "tail_fraction" not in normalized_book.columns:
        normalized_book["tail_fraction"] = np.nan
    if "tail_size" not in normalized_book.columns:
        normalized_book["tail_size"] = np.nan
    if "beta_mode" not in normalized_book.columns:
        normalized_book["beta_mode"] = (
            str(realized_forecasts["beta_mode"].iloc[0])
            if "beta_mode" in realized_forecasts.columns and not realized_forecasts.empty
            else "unknown"
        )

    realized = _build_realized_returns(realized_forecasts)
    metrics = BacktestRunner().run(normalized_book, realized, costs_bps=costs_bps)

    scoreboard_positions = _build_live_scoreboard_positions(
        baseline_book=normalized_book,
        baseline_position_returns=metrics.position_returns,
        forecasts=realized_forecasts,
        labeled_frame=labeled_frame,
    )
    scoreboard_daily = _build_live_scoreboard_daily(scoreboard_positions)

    score_column = "score_q50"
    if "ranking_score_column" in normalized_book.columns and normalized_book["ranking_score_column"].notna().any():
        candidate = str(normalized_book["ranking_score_column"].dropna().iloc[0])
        if candidate in realized_forecasts.columns:
            score_column = candidate
    rank_ic = compute_rank_ic_for_score(realized_forecasts, score_column)
    execution_outputs = generate_near_open_execution_analysis(
        book_frame=normalized_book,
        raw_execution_bars=raw_execution_bars,
        benchmark_ticker=benchmark_ticker,
        costs_bps=costs_bps,
        execution_mode=execution_mode,
    )

    return {
        "reconciled_forecasts": realized_forecasts,
        "live_position_returns": metrics.position_returns,
        "live_metrics_summary": metrics.summary,
        "live_daily_returns": metrics.daily_returns,
        "live_scoreboard_positions": scoreboard_positions,
        "live_scoreboard_daily": scoreboard_daily,
        "live_rank_ic": rank_ic,
        **execution_outputs,
    }


def _build_basket_contribution_reports(position_returns: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    if "basket_id" not in position_returns.columns:
        return pd.DataFrame(), pd.DataFrame()

    daily = (
        position_returns.groupby(["prediction_session", "basket_id", "side"], as_index=False)
        .agg(
            gross_pnl=("gross_position_return", "sum"),
            mean_realized_return=("realized_day_return", "mean"),
            position_count=("ticker", "size"),
        )
        .sort_values(["prediction_session", "basket_id", "side"])
        .reset_index(drop=True)
    )
    long_side = daily[daily["side"] == "long"].rename(
        columns={"gross_pnl": "long_gross_pnl", "mean_realized_return": "long_mean_return", "position_count": "long_position_count"}
    )
    short_side = daily[daily["side"] == "short"].rename(
        columns={"gross_pnl": "short_gross_pnl", "mean_realized_return": "short_mean_return", "position_count": "short_position_count"}
    )
    basket_daily = long_side.merge(
        short_side[["prediction_session", "basket_id", "short_gross_pnl", "short_mean_return", "short_position_count"]],
        on=["prediction_session", "basket_id"],
        how="outer",
        validate="one_to_one",
    ).sort_values(["prediction_session", "basket_id"])
    basket_daily["total_gross_pnl"] = basket_daily["long_gross_pnl"].fillna(0.0) + basket_daily["short_gross_pnl"].fillna(0.0)
    basket_daily["long_short_spread"] = basket_daily["long_mean_return"] - basket_daily["short_mean_return"]

    total_abs_pnl = float(basket_daily["total_gross_pnl"].abs().sum())
    basket_summary = (
        basket_daily.groupby("basket_id", as_index=False)
        .agg(
            mean_daily_gross_pnl=("total_gross_pnl", "mean"),
            total_gross_pnl=("total_gross_pnl", "sum"),
            mean_long_short_spread=("long_short_spread", "mean"),
            mean_long_return=("long_mean_return", "mean"),
            mean_short_return=("short_mean_return", "mean"),
            long_position_count=("long_position_count", "sum"),
            short_position_count=("short_position_count", "sum"),
        )
        .sort_values("basket_id")
        .reset_index(drop=True)
    )
    basket_summary["pnl_contribution_share"] = (
        basket_summary["total_gross_pnl"].abs() / total_abs_pnl if total_abs_pnl > 0 else 0.0
    )
    return basket_summary, basket_daily.reset_index(drop=True)


def _build_ticker_attribution_reports(position_returns: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    positions = position_returns.copy()
    total_prediction_days = int(positions["prediction_session"].nunique()) if not positions.empty else 0
    positions["hit"] = np.where(
        positions["side"] == "long",
        positions["realized_day_return"] > 0,
        positions["realized_day_return"] < 0,
    )
    positions["signed_realized_alpha"] = np.where(
        positions["side"] == "long",
        positions["realized_day_return"],
        -positions["realized_day_return"],
    )
    summary = (
        positions.groupby(["ticker", "basket_id"], as_index=False)
        .agg(
            days_in_book=("prediction_session", "nunique"),
            long_days=("side", lambda series: int((series == "long").sum())),
            short_days=("side", lambda series: int((series == "short").sum())),
            total_gross_pnl=("gross_position_return", "sum"),
            mean_gross_pnl=("gross_position_return", "mean"),
            mean_realized_return=("realized_day_return", "mean"),
            mean_abs_score=("ranking_score", lambda series: float(series.abs().mean())),
            mean_signed_realized_alpha=("signed_realized_alpha", "mean"),
            hit_rate=("hit", "mean"),
        )
        .sort_values(["total_gross_pnl", "ticker"], ascending=[False, True])
        .reset_index(drop=True)
    )
    summary["mean_spread_contribution"] = (
        summary["total_gross_pnl"] / total_prediction_days if total_prediction_days > 0 else 0.0
    )
    return summary, positions.sort_values(["prediction_session", "gross_position_return"], ascending=[True, False]).reset_index(drop=True)


def _build_live_scoreboard_positions(
    *,
    baseline_book: pd.DataFrame,
    baseline_position_returns: pd.DataFrame,
    forecasts: pd.DataFrame,
    labeled_frame: pd.DataFrame,
) -> pd.DataFrame:
    if baseline_book.empty or baseline_position_returns.empty:
        return pd.DataFrame(
            columns=[
                "signal_date",
                "prediction_date",
                "signal_session",
                "prediction_session",
                "ticker",
                "basket_id",
                "side",
                "rank",
                "score_z",
                "score_plain",
                "ranking_score_column",
                "ranking_score",
                "book_mode",
                "tail_fraction",
                "tail_size",
                "beta_mode",
                "in_tail_book",
                "next_day_realized_return",
                "next_day_realized_alpha",
                "weight",
                "gross_position_return",
                "book_helped",
                "book_impact",
            ]
        )

    scoreboard = _normalize_time_columns(
        baseline_position_returns,
        ["signal_session", "prediction_session"],
    ).copy()
    raw_daily = _build_raw_daily_returns(labeled_frame)
    scoreboard = scoreboard.merge(
        raw_daily,
        on=["ticker", "prediction_session"],
        how="left",
        validate="many_to_one",
    )

    if "score_plain" not in scoreboard.columns or "score_z" not in scoreboard.columns:
        available_scores = [
            column
            for column in ["score_plain", "score_z"]
            if column in forecasts.columns and column not in scoreboard.columns
        ]
        if available_scores:
            scoreboard = scoreboard.merge(
                forecasts[["ticker", "signal_session", "prediction_session", *available_scores]].drop_duplicates(
                    subset=["ticker", "signal_session", "prediction_session"]
                ),
                on=["ticker", "signal_session", "prediction_session"],
                how="left",
                validate="many_to_one",
            )

    scoreboard["signal_date"] = scoreboard["signal_session"].dt.strftime("%Y-%m-%d")
    scoreboard["prediction_date"] = scoreboard["prediction_session"].dt.strftime("%Y-%m-%d")
    scoreboard["in_tail_book"] = True
    scoreboard["next_day_realized_alpha"] = scoreboard["realized_day_return"]
    scoreboard["book_helped"] = scoreboard["gross_position_return"] > 0
    scoreboard["book_impact"] = np.where(
        scoreboard["gross_position_return"] > 0,
        "helped",
        np.where(scoreboard["gross_position_return"] < 0, "hurt", "flat"),
    )

    columns = [
        "signal_date",
        "prediction_date",
        "signal_session",
        "prediction_session",
        "ticker",
        "basket_id",
        "side",
        "rank",
        "score_z",
        "score_plain",
        "ranking_score_column",
        "ranking_score",
        "book_mode",
        "tail_fraction",
        "tail_size",
        "beta_mode",
        "in_tail_book",
        "next_day_realized_return",
        "next_day_realized_alpha",
        "weight",
        "gross_position_return",
        "book_helped",
        "book_impact",
    ]
    for column in columns:
        if column not in scoreboard.columns:
            scoreboard[column] = np.nan
    return scoreboard[columns].sort_values(["prediction_session", "side", "rank", "ticker"]).reset_index(drop=True)


def _build_live_scoreboard_daily(scoreboard_positions: pd.DataFrame) -> pd.DataFrame:
    if scoreboard_positions.empty:
        return pd.DataFrame(
            columns=[
                "signal_date",
                "prediction_date",
                "signal_session",
                "prediction_session",
                "long_basket_return",
                "short_basket_return",
                "long_short_spread",
                "gross_long_short_pnl",
                "book_hit",
                "best_contributor",
                "best_contributor_pnl",
                "worst_contributor",
                "worst_contributor_pnl",
                "growth_result_gross_pnl",
                "defensive_result_gross_pnl",
            ]
        )

    positions = _normalize_time_columns(scoreboard_positions, ["signal_session", "prediction_session"]).copy()
    side_daily = (
        positions.groupby(["prediction_session", "side"], as_index=False)
        .agg(
            signal_session=("signal_session", "first"),
            long_or_short_alpha=("next_day_realized_alpha", "mean"),
            side_pnl=("gross_position_return", "sum"),
        )
        .sort_values(["prediction_session", "side"])
        .reset_index(drop=True)
    )
    long_side = side_daily[side_daily["side"] == "long"].rename(
        columns={"long_or_short_alpha": "long_basket_return", "side_pnl": "long_side_pnl"}
    )
    short_side = side_daily[side_daily["side"] == "short"].rename(
        columns={"long_or_short_alpha": "short_basket_return", "side_pnl": "short_side_pnl"}
    )
    daily = long_side.merge(
        short_side[["prediction_session", "short_basket_return", "short_side_pnl"]],
        on="prediction_session",
        how="outer",
        validate="one_to_one",
    ).sort_values("prediction_session")
    daily["signal_date"] = daily["signal_session"].dt.strftime("%Y-%m-%d")
    daily["prediction_date"] = daily["prediction_session"].dt.strftime("%Y-%m-%d")
    daily["long_short_spread"] = daily["long_basket_return"] - daily["short_basket_return"]
    daily["gross_long_short_pnl"] = daily["long_side_pnl"].fillna(0.0) + daily["short_side_pnl"].fillna(0.0)
    daily["book_hit"] = daily["gross_long_short_pnl"] > 0

    best = (
        positions.sort_values(["prediction_session", "gross_position_return", "ticker"], ascending=[True, False, True])
        .groupby("prediction_session", as_index=False)
        .first()[["prediction_session", "ticker", "gross_position_return"]]
        .rename(columns={"ticker": "best_contributor", "gross_position_return": "best_contributor_pnl"})
    )
    worst = (
        positions.sort_values(["prediction_session", "gross_position_return", "ticker"], ascending=[True, True, True])
        .groupby("prediction_session", as_index=False)
        .first()[["prediction_session", "ticker", "gross_position_return"]]
        .rename(columns={"ticker": "worst_contributor", "gross_position_return": "worst_contributor_pnl"})
    )
    basket_daily = (
        positions.groupby(["prediction_session", "basket_id"], as_index=False)
        .agg(basket_gross_pnl=("gross_position_return", "sum"))
        .pivot(index="prediction_session", columns="basket_id", values="basket_gross_pnl")
        .rename(
            columns={
                "growth": "growth_result_gross_pnl",
                "defensive_value": "defensive_result_gross_pnl",
            }
        )
        .reset_index()
    )
    daily = daily.merge(best, on="prediction_session", how="left", validate="one_to_one")
    daily = daily.merge(worst, on="prediction_session", how="left", validate="one_to_one")
    daily = daily.merge(basket_daily, on="prediction_session", how="left", validate="one_to_one")
    columns = [
        "signal_date",
        "prediction_date",
        "signal_session",
        "prediction_session",
        "long_basket_return",
        "short_basket_return",
        "long_short_spread",
        "gross_long_short_pnl",
        "book_hit",
        "best_contributor",
        "best_contributor_pnl",
        "worst_contributor",
        "worst_contributor_pnl",
        "growth_result_gross_pnl",
        "defensive_result_gross_pnl",
    ]
    return daily[columns].reset_index(drop=True)


def _format_top_tickers(ticker_attribution_summary: pd.DataFrame, *, ascending: bool, limit: int = 5) -> str:
    if ticker_attribution_summary.empty:
        return ""
    ordered = ticker_attribution_summary.sort_values(["total_gross_pnl", "ticker"], ascending=[ascending, True]).head(limit)
    return " | ".join(
        f"{row.ticker}:{row.total_gross_pnl:.6f}"
        for row in ordered.itertuples(index=False)
    )


def _build_oos_window_reports(
    *,
    forecasts: pd.DataFrame,
    labeled_frame: pd.DataFrame,
    live_scoreboard_daily: pd.DataFrame,
    baseline_rank_ic: pd.DataFrame,
    slots_per_session: int = 7,
    target_days: tuple[int, ...] = (20, 30),
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    normalized_forecasts = _normalize_time_columns(forecasts, ["signal_session", "prediction_session"])
    normalized_labeled = _normalize_time_columns(labeled_frame, ["session_date", "timestamp"])
    normalized_daily = _normalize_time_columns(live_scoreboard_daily, ["signal_session", "prediction_session"])

    context_values = (
        pd.to_numeric(normalized_forecasts["context_length"], errors="coerce").dropna()
        if "context_length" in normalized_forecasts.columns
        else pd.Series(dtype=float)
    )
    context_length = float(context_values.iloc[0]) if not context_values.empty else float("nan")
    required_context_sessions = (
        int(ceil(context_length / slots_per_session))
        if not np.isnan(context_length)
        else float("nan")
    )
    complete_sessions = (
        int(pd.to_datetime(normalized_labeled["session_date"], utc=True, errors="coerce").dropna().nunique())
        if "session_date" in normalized_labeled.columns
        else 0
    )
    current_prediction_days = int(normalized_forecasts["prediction_session"].nunique()) if not normalized_forecasts.empty else 0
    theoretical_max_prediction_days = (
        max(0, complete_sessions - int(required_context_sessions))
        if not np.isnan(required_context_sessions)
        else float("nan")
    )

    session_dates = (
        pd.to_datetime(normalized_labeled["session_date"], utc=True, errors="coerce").dropna().sort_values().unique().tolist()
        if "session_date" in normalized_labeled.columns
        else []
    )
    summary_record: dict[str, object] = {
        "complete_sessions": complete_sessions,
        "slots_per_session": slots_per_session,
        "context_length": context_length,
        "required_context_sessions": required_context_sessions,
        "current_prediction_days": current_prediction_days,
        "theoretical_max_prediction_days": theoretical_max_prediction_days,
        "session_start": session_dates[0] if session_dates else pd.NaT,
        "session_end": session_dates[-1] if session_dates else pd.NaT,
    }
    for target_day in target_days:
        summary_record[f"target_gap_{target_day}d"] = max(0, target_day - current_prediction_days)
    summary = pd.DataFrame([summary_record])

    rank_ic_daily = baseline_rank_ic[baseline_rank_ic["prediction_session"].notna()].copy()
    if not rank_ic_daily.empty:
        rank_ic_daily = _normalize_time_columns(rank_ic_daily, ["prediction_session"])[["prediction_session", "rank_ic"]]
    stability_daily = normalized_daily.merge(
        rank_ic_daily,
        on="prediction_session",
        how="left",
        validate="one_to_one",
    ) if not normalized_daily.empty else pd.DataFrame()
    if stability_daily.empty:
        stability_daily = pd.DataFrame(
            columns=[
                "signal_session",
                "prediction_session",
                "signal_date",
                "prediction_date",
                "long_basket_return",
                "short_basket_return",
                "long_short_spread",
                "gross_long_short_pnl",
                "book_hit",
                "rank_ic",
                "cumulative_gross_long_short_pnl",
                "cumulative_book_hit_rate",
                "prediction_week",
            ]
        )
    else:
        stability_daily = stability_daily.sort_values("prediction_session").reset_index(drop=True)
        stability_daily["cumulative_gross_long_short_pnl"] = stability_daily["gross_long_short_pnl"].fillna(0.0).cumsum()
        stability_daily["cumulative_book_hit_rate"] = stability_daily["book_hit"].fillna(False).astype(float).expanding().mean()
        iso_calendar = stability_daily["prediction_session"].dt.isocalendar()
        stability_daily["prediction_week"] = (
            iso_calendar["year"].astype(str) + "-W" + iso_calendar["week"].astype(str).str.zfill(2)
        )

    weekly = pd.DataFrame()
    if not stability_daily.empty:
        weekly = (
            stability_daily.groupby("prediction_week", as_index=False)
            .agg(
                prediction_days=("prediction_session", "nunique"),
                mean_daily_gross_pnl=("gross_long_short_pnl", "mean"),
                total_gross_pnl=("gross_long_short_pnl", "sum"),
                mean_daily_long_short_spread=("long_short_spread", "mean"),
                mean_rank_ic=("rank_ic", "mean"),
                positive_rank_ic_rate=("rank_ic", lambda series: float((series > 0).mean())),
                book_hit_rate=("book_hit", lambda series: float(series.astype(float).mean())),
            )
            .sort_values("prediction_week")
            .reset_index(drop=True)
        )

    if stability_daily.empty:
        concentration = pd.DataFrame(
            [
                {
                    "prediction_days": 0,
                    "positive_days": 0,
                    "negative_days": 0,
                    "best_day_abs_pnl_share": float("nan"),
                    "top2_abs_pnl_share": float("nan"),
                    "best_day_prediction_session": pd.NaT,
                    "worst_day_prediction_session": pd.NaT,
                }
            ]
        )
    else:
        pnl = stability_daily["gross_long_short_pnl"].fillna(0.0)
        abs_pnl = pnl.abs()
        total_abs_pnl = float(abs_pnl.sum())
        ordered_abs = abs_pnl.sort_values(ascending=False).tolist()
        best_abs_share = float(ordered_abs[0] / total_abs_pnl) if ordered_abs and total_abs_pnl > 0 else float("nan")
        top2_abs_share = float(sum(ordered_abs[:2]) / total_abs_pnl) if ordered_abs and total_abs_pnl > 0 else float("nan")
        best_day_index = pnl.idxmax()
        worst_day_index = pnl.idxmin()
        concentration = pd.DataFrame(
            [
                {
                    "prediction_days": int(stability_daily["prediction_session"].nunique()),
                    "positive_days": int((pnl > 0).sum()),
                    "negative_days": int((pnl < 0).sum()),
                    "flat_days": int((pnl == 0).sum()),
                    "total_gross_pnl": float(pnl.sum()),
                    "total_abs_gross_pnl": total_abs_pnl,
                    "best_day_gross_pnl": float(pnl.max()),
                    "worst_day_gross_pnl": float(pnl.min()),
                    "best_day_abs_pnl_share": best_abs_share,
                    "top2_abs_pnl_share": top2_abs_share,
                    "best_day_prediction_session": stability_daily.loc[best_day_index, "prediction_session"],
                    "worst_day_prediction_session": stability_daily.loc[worst_day_index, "prediction_session"],
                }
            ]
        )

    return summary, stability_daily, weekly, concentration


def _build_side_isolation_reports(position_returns: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if position_returns.empty:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    positions = position_returns.copy()
    positions["hit"] = np.where(
        positions["side"] == "long",
        positions["realized_day_return"] > 0,
        positions["realized_day_return"] < 0,
    )
    positions["signed_realized_alpha"] = np.where(
        positions["side"] == "long",
        positions["realized_day_return"],
        -positions["realized_day_return"],
    )

    daily = (
        positions.groupby(["prediction_session", "side"], as_index=False)
        .agg(
            signal_session=("signal_session", "first"),
            basket_id_count=("basket_id", "nunique"),
            position_count=("ticker", "size"),
            mean_realized_return=("realized_day_return", "mean"),
            gross_pnl=("gross_position_return", "sum"),
            hit_rate=("hit", "mean"),
        )
        .sort_values(["prediction_session", "side"])
        .reset_index(drop=True)
    )

    summary_records: list[dict[str, object]] = []
    ticker_frames: list[pd.DataFrame] = []
    for side, side_positions in positions.groupby("side", sort=True):
        side_daily = daily[daily["side"] == side].copy()
        total_abs_pnl = float(side_positions["gross_position_return"].abs().sum())
        ticker_summary = (
            side_positions.groupby(["side", "ticker", "basket_id"], as_index=False)
            .agg(
                days_in_book=("prediction_session", "nunique"),
                total_gross_pnl=("gross_position_return", "sum"),
                mean_gross_pnl=("gross_position_return", "mean"),
                mean_realized_return=("realized_day_return", "mean"),
                mean_abs_score=("ranking_score", lambda series: float(series.abs().mean())),
                mean_signed_realized_alpha=("signed_realized_alpha", "mean"),
                hit_rate=("hit", "mean"),
            )
            .sort_values(["total_gross_pnl", "ticker"], ascending=[False, True])
            .reset_index(drop=True)
        )
        ticker_summary["abs_pnl_share"] = (
            ticker_summary["total_gross_pnl"].abs() / total_abs_pnl
            if total_abs_pnl > 0
            else 0.0
        )
        ticker_frames.append(ticker_summary)

        ordered_abs = ticker_summary["abs_pnl_share"].sort_values(ascending=False).tolist()
        top_ticker = ticker_summary.iloc[0]["ticker"] if not ticker_summary.empty else ""
        top_ticker_pnl = float(ticker_summary.iloc[0]["total_gross_pnl"]) if not ticker_summary.empty else float("nan")
        summary_records.append(
            {
                "side": side,
                "prediction_days": int(side_daily["prediction_session"].nunique()),
                "position_count": int(len(side_positions)),
                "positive_days": int((side_daily["gross_pnl"] > 0).sum()),
                "negative_days": int((side_daily["gross_pnl"] < 0).sum()),
                "mean_daily_gross_pnl": float(side_daily["gross_pnl"].mean()) if not side_daily.empty else float("nan"),
                "median_daily_gross_pnl": float(side_daily["gross_pnl"].median()) if not side_daily.empty else float("nan"),
                "mean_daily_realized_return": float(side_daily["mean_realized_return"].mean()) if not side_daily.empty else float("nan"),
                "hit_rate": float(side_positions["hit"].mean()),
                "total_gross_pnl": float(side_positions["gross_position_return"].sum()),
                "top_ticker": top_ticker,
                "top_ticker_total_gross_pnl": top_ticker_pnl,
                "top_ticker_abs_pnl_share": float(ordered_abs[0]) if ordered_abs else float("nan"),
                "top2_ticker_abs_pnl_share": float(sum(ordered_abs[:2])) if ordered_abs else float("nan"),
            }
        )

    summary = pd.DataFrame.from_records(summary_records).sort_values("side").reset_index(drop=True)
    ticker_summary = (
        pd.concat(ticker_frames, ignore_index=True, sort=False)
        if ticker_frames
        else pd.DataFrame()
    )
    return summary, daily, ticker_summary


def _coerce_short_available(value: object) -> bool:
    if pd.isna(value):
        return True
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (int, float)) and not pd.isna(value):
        return bool(int(value))
    normalized = str(value).strip().lower()
    if normalized in {"true", "t", "yes", "y", "1", "available", "easy", "etb"}:
        return True
    if normalized in {"false", "f", "no", "n", "0", "unavailable", "hard", "htb"}:
        return False
    return True


def _build_borrow_overlay_reports(
    *,
    position_returns: pd.DataFrame,
    costs_bps: tuple[int, ...],
    default_borrow_fee_annual_bps: float,
    borrow_profile: pd.DataFrame | None = None,
    unavailable_tickers: tuple[str, ...] = (),
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if position_returns.empty:
        empty = pd.DataFrame()
        return pd.DataFrame(), empty, empty, empty

    unavailable_set = {ticker.strip().upper() for ticker in unavailable_tickers if str(ticker).strip()}
    profile = pd.DataFrame(columns=["ticker", "borrow_fee_annual_bps", "short_available"])
    if borrow_profile is not None and not borrow_profile.empty:
        profile = borrow_profile.copy()
        if "ticker" not in profile.columns:
            raise ValueError("Borrow profile must contain a `ticker` column.")
        profile["ticker"] = profile["ticker"].astype(str).str.upper().str.strip()
        if "borrow_fee_annual_bps" not in profile.columns:
            profile["borrow_fee_annual_bps"] = default_borrow_fee_annual_bps
        profile["borrow_fee_annual_bps"] = pd.to_numeric(profile["borrow_fee_annual_bps"], errors="coerce").fillna(
            default_borrow_fee_annual_bps
        )
        if "short_available" not in profile.columns:
            profile["short_available"] = True
        profile["short_available"] = profile["short_available"].apply(_coerce_short_available)
        profile = profile[["ticker", "borrow_fee_annual_bps", "short_available"]].drop_duplicates(subset=["ticker"], keep="last")

    overlay = position_returns.copy()
    overlay["ticker"] = overlay["ticker"].astype(str).str.upper()
    overlay = overlay.merge(profile, on="ticker", how="left", validate="many_to_one")
    overlay["borrow_fee_annual_bps"] = pd.to_numeric(overlay["borrow_fee_annual_bps"], errors="coerce").fillna(
        default_borrow_fee_annual_bps
    )
    overlay["short_available"] = overlay["short_available"].apply(_coerce_short_available)
    if unavailable_set:
        overlay.loc[overlay["ticker"].isin(unavailable_set), "short_available"] = False

    short_mask = overlay["side"] == "short"
    overlay["borrow_fee_daily_rate"] = 0.0
    overlay.loc[short_mask, "borrow_fee_daily_rate"] = overlay.loc[short_mask, "borrow_fee_annual_bps"] / 10000.0 / 252.0
    overlay["borrow_excluded"] = short_mask & (~overlay["short_available"])
    overlay["executed_weight_after_borrow"] = np.where(overlay["borrow_excluded"], 0.0, overlay["weight"])
    overlay["borrow_cost"] = np.where(
        short_mask & (~overlay["borrow_excluded"]),
        overlay["executed_weight_after_borrow"].abs() * overlay["borrow_fee_daily_rate"],
        0.0,
    )
    overlay["gross_position_return_after_borrow"] = np.where(
        overlay["borrow_excluded"],
        0.0,
        overlay["gross_position_return"] - overlay["borrow_cost"],
    )

    daily = (
        overlay.groupby("prediction_session", as_index=False)
        .agg(
            gross_return_after_borrow=("gross_position_return_after_borrow", "sum"),
            gross_exposure_after_borrow=("executed_weight_after_borrow", lambda series: float(series.abs().sum())),
            borrow_cost=("borrow_cost", "sum"),
            excluded_short_positions=("borrow_excluded", "sum"),
            excluded_short_tickers=("ticker", lambda series: 0),
        )
        .sort_values("prediction_session")
        .reset_index(drop=True)
    )
    if not overlay.empty:
        excluded_by_day = (
            overlay[overlay["borrow_excluded"]]
            .groupby("prediction_session", as_index=False)["ticker"]
            .nunique()
            .rename(columns={"ticker": "excluded_short_tickers"})
        )
        if not excluded_by_day.empty:
            daily = daily.drop(columns=["excluded_short_tickers"]).merge(
                excluded_by_day,
                on="prediction_session",
                how="left",
                validate="one_to_one",
            )
            daily["excluded_short_tickers"] = daily["excluded_short_tickers"].fillna(0).astype(int)

    for cost_bps in costs_bps:
        cost_rate = cost_bps / 10000.0
        daily[f"net_return_after_borrow_{cost_bps}bps"] = (
            daily["gross_return_after_borrow"] - (daily["gross_exposure_after_borrow"] * cost_rate)
        )

    assumptions = pd.DataFrame(
        [
            {
                "default_borrow_fee_annual_bps": float(default_borrow_fee_annual_bps),
                "borrow_profile_rows": int(len(profile)),
                "unavailable_ticker_count": int(len(unavailable_set)),
                "unavailable_tickers": ",".join(sorted(unavailable_set)),
                "overlay_mode": "basic_short_borrow_overlay",
                "availability_treatment": "exclude_unavailable_shorts_without_reweighting",
                "fee_treatment": "subtract_daily_borrow_fee_from_executed_short_positions",
            }
        ]
    )

    short_positions = overlay[short_mask].copy()
    summary = pd.DataFrame(
        [
            {
                "prediction_days": int(daily["prediction_session"].nunique()),
                "gross_sharpe": float(_annualized_sharpe(daily["gross_return_after_borrow"])),
                "net_sharpe_10bps": float(_annualized_sharpe(daily["net_return_after_borrow_10bps"])) if "net_return_after_borrow_10bps" in daily.columns else float("nan"),
                "net_sharpe_20bps": float(_annualized_sharpe(daily["net_return_after_borrow_20bps"])) if "net_return_after_borrow_20bps" in daily.columns else float("nan"),
                "net_sharpe_40bps": float(_annualized_sharpe(daily["net_return_after_borrow_40bps"])) if "net_return_after_borrow_40bps" in daily.columns else float("nan"),
                "mean_daily_gross_return_after_borrow": float(daily["gross_return_after_borrow"].mean()),
                "mean_daily_borrow_cost": float(daily["borrow_cost"].mean()),
                "excluded_short_positions": int(short_positions["borrow_excluded"].sum()),
                "excluded_prediction_days": int(short_positions.loc[short_positions["borrow_excluded"], "prediction_session"].nunique()),
                "mean_short_borrow_fee_annual_bps": float(short_positions["borrow_fee_annual_bps"].mean()) if not short_positions.empty else float("nan"),
                "total_borrow_cost": float(short_positions["borrow_cost"].sum()) if not short_positions.empty else 0.0,
                "short_available_rate": float((~short_positions["borrow_excluded"]).mean()) if not short_positions.empty else float("nan"),
            }
        ]
    )

    ticker_summary = pd.DataFrame()
    if not short_positions.empty:
        ticker_summary = (
            short_positions.groupby(["ticker", "basket_id"], as_index=False)
            .agg(
                short_days=("prediction_session", "nunique"),
                excluded_short_days=("borrow_excluded", "sum"),
                mean_borrow_fee_annual_bps=("borrow_fee_annual_bps", "mean"),
                total_borrow_cost=("borrow_cost", "sum"),
                total_gross_pnl_before_borrow=("gross_position_return", "sum"),
                total_gross_pnl_after_borrow=("gross_position_return_after_borrow", "sum"),
                short_available_rate=("borrow_excluded", lambda series: float((~series.astype(bool)).mean())),
            )
            .sort_values(["total_gross_pnl_after_borrow", "ticker"], ascending=[False, True])
            .reset_index(drop=True)
        )
        ticker_summary["borrow_pnl_drag"] = (
            ticker_summary["total_gross_pnl_before_borrow"] - ticker_summary["total_gross_pnl_after_borrow"]
        )

    return assumptions, summary, daily, overlay.sort_values(["prediction_session", "side", "rank", "ticker"]).reset_index(drop=True), ticker_summary


def build_backtest_scorecard(
    *,
    forecasts: pd.DataFrame,
    signal_outputs: dict[str, pd.DataFrame],
    execution_outputs: dict[str, pd.DataFrame],
) -> pd.DataFrame:
    baseline_summary = signal_outputs.get("signal_execution_baseline_summary", pd.DataFrame())
    score_behavior = signal_outputs.get("score_behavior_summary", pd.DataFrame())
    basket_book_comparison = signal_outputs.get("basket_book_comparison", pd.DataFrame())
    side_book_comparison = signal_outputs.get("side_book_comparison", pd.DataFrame())
    ticker_attribution_summary = signal_outputs.get("ticker_attribution_summary", pd.DataFrame())
    live_scoreboard_daily = signal_outputs.get("live_scoreboard_daily", pd.DataFrame())
    oos_window_summary = signal_outputs.get("oos_window_summary", pd.DataFrame())
    oos_concentration_summary = signal_outputs.get("oos_concentration_summary", pd.DataFrame())
    side_isolation_summary = signal_outputs.get("side_isolation_summary", pd.DataFrame())
    borrow_overlay_summary = signal_outputs.get("borrow_overlay_summary", pd.DataFrame())
    execution_sensitivity = execution_outputs.get("execution_sensitivity_comparison", pd.DataFrame())

    if baseline_summary.empty:
        return pd.DataFrame()

    baseline_row = baseline_summary.iloc[0]
    score_behavior_lookup = score_behavior.set_index("score_column") if not score_behavior.empty else pd.DataFrame()
    basket_lookup = (
        basket_book_comparison.set_index("basket_slice")
        if not basket_book_comparison.empty and "basket_slice" in basket_book_comparison.columns
        else pd.DataFrame()
    )
    side_lookup = (
        side_book_comparison.set_index("side_slice")
        if not side_book_comparison.empty and "side_slice" in side_book_comparison.columns
        else pd.DataFrame()
    )
    side_isolation_lookup = (
        side_isolation_summary.set_index("side")
        if not side_isolation_summary.empty and "side" in side_isolation_summary.columns
        else pd.DataFrame()
    )
    execution_lookup = (
        execution_sensitivity.set_index("execution_mode")
        if not execution_sensitivity.empty and "execution_mode" in execution_sensitivity.columns
        else pd.DataFrame()
    )

    def _lookup(frame: pd.DataFrame, row_key: str, column: str) -> float:
        if frame.empty or row_key not in frame.index or column not in frame.columns:
            return float("nan")
        return float(frame.loc[row_key, column])

    oos_row = oos_window_summary.iloc[0] if not oos_window_summary.empty else pd.Series(dtype=object)
    oos_concentration_row = oos_concentration_summary.iloc[0] if not oos_concentration_summary.empty else pd.Series(dtype=object)
    borrow_row = borrow_overlay_summary.iloc[0] if not borrow_overlay_summary.empty else pd.Series(dtype=object)
    book_hit_rate = float(live_scoreboard_daily["book_hit"].mean()) if not live_scoreboard_daily.empty else float("nan")
    row = {
        "universe_size": int(forecasts["ticker"].nunique()) if not forecasts.empty else 0,
        "prediction_days": int(baseline_row["prediction_days"]),
        "ranking_score_column": str(baseline_row["score_column"]),
        "mean_rank_ic": float(baseline_row["mean_rank_ic"]),
        "median_rank_ic": float(baseline_row["median_rank_ic"]),
        "positive_rank_ic_rate": float(baseline_row["positive_rank_ic_rate"]),
        "gross_sharpe": float(baseline_row["gross_sharpe"]),
        "net_sharpe_10bps": float(baseline_row["net_sharpe_10bps"]),
        "net_sharpe_20bps": float(baseline_row["net_sharpe_20bps"]),
        "net_sharpe_40bps": float(baseline_row["net_sharpe_40bps"]),
        "long_hit_rate": float(baseline_row["long_hit_rate"]),
        "short_hit_rate": float(baseline_row["short_hit_rate"]),
        "book_hit_rate": book_hit_rate,
        "top1_capture_mean": float(baseline_row["top1_capture_mean"]),
        "top5_capture_mean": float(baseline_row["top5_capture_mean"]),
        "mean_daily_long_return": float(baseline_row["mean_daily_long_return"]),
        "mean_daily_short_return": float(baseline_row["mean_daily_short_return"]),
        "mean_daily_long_short_spread": float(baseline_row["mean_daily_long_short_spread"]),
        "score_plain_sign_accuracy": _lookup(score_behavior_lookup, "score_plain", "sign_accuracy"),
        "score_z_sign_accuracy": _lookup(score_behavior_lookup, "score_z", "sign_accuracy"),
        "combined_net_sharpe_10bps": _lookup(basket_lookup, "combined", "net_sharpe_10bps"),
        "growth_only_net_sharpe_10bps": _lookup(basket_lookup, "growth_only", "net_sharpe_10bps"),
        "defensive_only_net_sharpe_10bps": _lookup(basket_lookup, "defensive_only", "net_sharpe_10bps"),
        "combined_mean_daily_return": _lookup(basket_lookup, "combined", "mean_daily_return"),
        "growth_only_mean_daily_return": _lookup(basket_lookup, "growth_only", "mean_daily_return"),
        "defensive_only_mean_daily_return": _lookup(basket_lookup, "defensive_only", "mean_daily_return"),
        "long_only_net_sharpe_10bps": _lookup(side_lookup, "long_only", "net_sharpe_10bps"),
        "short_only_net_sharpe_10bps": _lookup(side_lookup, "short_only", "net_sharpe_10bps"),
        "long_short_net_sharpe_10bps": _lookup(side_lookup, "long_short", "net_sharpe_10bps"),
        "exact_open_net_sharpe_10bps": _lookup(execution_lookup, "exact_open", "net_sharpe_10bps"),
        "exact_open_net_sharpe_20bps": _lookup(execution_lookup, "exact_open", "net_sharpe_20bps"),
        "exact_open_net_sharpe_40bps": _lookup(execution_lookup, "exact_open", "net_sharpe_40bps"),
        "first_5m_avg_net_sharpe_10bps": _lookup(execution_lookup, "first_5m_avg", "net_sharpe_10bps"),
        "first_5m_avg_net_sharpe_20bps": _lookup(execution_lookup, "first_5m_avg", "net_sharpe_20bps"),
        "first_5m_avg_net_sharpe_40bps": _lookup(execution_lookup, "first_5m_avg", "net_sharpe_40bps"),
        "first_15m_avg_net_sharpe_10bps": _lookup(execution_lookup, "first_15m_avg", "net_sharpe_10bps"),
        "first_15m_avg_net_sharpe_20bps": _lookup(execution_lookup, "first_15m_avg", "net_sharpe_20bps"),
        "first_15m_avg_net_sharpe_40bps": _lookup(execution_lookup, "first_15m_avg", "net_sharpe_40bps"),
        "oos_complete_sessions": oos_row.get("complete_sessions", float("nan")),
        "oos_required_context_sessions": oos_row.get("required_context_sessions", float("nan")),
        "oos_theoretical_max_prediction_days": oos_row.get("theoretical_max_prediction_days", float("nan")),
        "oos_target_gap_20d": oos_row.get("target_gap_20d", float("nan")),
        "oos_target_gap_30d": oos_row.get("target_gap_30d", float("nan")),
        "best_day_abs_pnl_share": oos_concentration_row.get("best_day_abs_pnl_share", float("nan")),
        "top2_abs_pnl_share": oos_concentration_row.get("top2_abs_pnl_share", float("nan")),
        "long_side_top_ticker_abs_pnl_share": _lookup(side_isolation_lookup, "long", "top_ticker_abs_pnl_share"),
        "short_side_top_ticker_abs_pnl_share": _lookup(side_isolation_lookup, "short", "top_ticker_abs_pnl_share"),
        "borrow_overlay_net_sharpe_10bps": borrow_row.get("net_sharpe_10bps", float("nan")),
        "borrow_overlay_net_sharpe_20bps": borrow_row.get("net_sharpe_20bps", float("nan")),
        "borrow_overlay_net_sharpe_40bps": borrow_row.get("net_sharpe_40bps", float("nan")),
        "borrow_overlay_excluded_short_positions": borrow_row.get("excluded_short_positions", float("nan")),
        "borrow_overlay_excluded_prediction_days": borrow_row.get("excluded_prediction_days", float("nan")),
        "borrow_overlay_mean_daily_borrow_cost": borrow_row.get("mean_daily_borrow_cost", float("nan")),
        "borrow_overlay_mean_short_fee_annual_bps": borrow_row.get("mean_short_borrow_fee_annual_bps", float("nan")),
        "top_contributors": _format_top_tickers(ticker_attribution_summary, ascending=False),
        "top_drags": _format_top_tickers(ticker_attribution_summary, ascending=True),
        "execution_mode_label": STRICT_EXECUTION_MODE_LABEL,
        "exact_open_entry_formula": EXACT_OPEN_ENTRY_FORMULA,
        "first_5m_avg_entry_formula": FIRST_5M_ENTRY_FORMULA,
        "first_15m_avg_entry_formula": FIRST_15M_ENTRY_FORMULA,
        "execution_exit_formula": EXECUTION_EXIT_FORMULA,
        "execution_alpha_formula": EXECUTION_ALPHA_FORMULA,
    }
    return pd.DataFrame([row])


def generate_v2_diagnostics(
    forecasts: pd.DataFrame,
    *,
    data_quality_events: pd.DataFrame | None = None,
    data_quality_summary: pd.DataFrame | None = None,
    labeled_frame: pd.DataFrame | None = None,
    benchmark_ticker: str = "SPY",
    n_long: int = 5,
    n_short: int = 5,
    costs_bps: tuple[int, ...] = (10, 20, 40),
    baseline_score_column: str = "score_z",
    score_columns: tuple[str, ...] = ("score_plain", "score_z", "score_risk"),
) -> dict[str, pd.DataFrame]:
    normalized_forecasts = _normalize_time_columns(forecasts, ["signal_session", "prediction_session"])
    available_scores = [column for column in score_columns if column in normalized_forecasts.columns]

    summary_rows: list[pd.DataFrame] = []
    daily_rows: list[pd.DataFrame] = []
    rank_ic_rows: list[pd.DataFrame] = []
    books: dict[str, pd.DataFrame] = {}

    for score_column in available_scores:
        summary_row, daily_side, book_frame, rank_ic = _build_score_summary(
            forecasts=normalized_forecasts,
            score_column=score_column,
            n_long=n_long,
            n_short=n_short,
            costs_bps=costs_bps,
        )
        summary_rows.append(summary_row)
        daily_rows.append(daily_side.assign(score_column=score_column))
        rank_ic_rows.append(rank_ic.assign(score_column=score_column))
        books[score_column] = book_frame

    ranking_signal_comparison = pd.concat(summary_rows, ignore_index=True, sort=False)
    ranking_signal_daily = pd.concat(daily_rows, ignore_index=True, sort=False)
    ranking_signal_rank_ic = pd.concat(rank_ic_rows, ignore_index=True, sort=False)

    baseline_summary = ranking_signal_comparison[ranking_signal_comparison["score_column"] == baseline_score_column].reset_index(drop=True)
    score_relationships = _build_score_relationships(normalized_forecasts)
    beta_summary, beta_by_ticker = _build_beta_summary(normalized_forecasts)
    data_quality_overview, data_quality_by_basket, data_quality_worst_tickers = _build_data_quality_reports(
        normalized_forecasts,
        data_quality_events,
        data_quality_summary,
    )
    beta_effect_comparison = _build_beta_effect_comparison(
        normalized_forecasts,
        labeled_frame,
        benchmark_ticker=benchmark_ticker,
        n_long=n_long,
        n_short=n_short,
        costs_bps=costs_bps,
    )

    outputs = {
        "baseline_summary": baseline_summary,
        "ranking_signal_comparison": ranking_signal_comparison,
        "ranking_signal_daily": ranking_signal_daily,
        "ranking_signal_rank_ic": ranking_signal_rank_ic,
        "score_relationships": score_relationships,
        "beta_summary": beta_summary,
        "beta_by_ticker": beta_by_ticker,
        "beta_effect_comparison": beta_effect_comparison,
        "data_quality_overview": data_quality_overview,
        "data_quality_by_basket": data_quality_by_basket,
        "data_quality_worst_tickers": data_quality_worst_tickers,
    }
    for score_column, book_frame in books.items():
        outputs[f"book_{score_column}"] = book_frame
    return outputs


def generate_signal_execution_analysis(
    forecasts: pd.DataFrame,
    *,
    labeled_frame: pd.DataFrame,
    execution_score_column: str = "score_z",
    score_behavior_columns: tuple[str, ...] = ("score_plain", "score_z"),
    benchmark_ticker: str = "SPY",
    n_long: int = 5,
    n_short: int = 5,
    costs_bps: tuple[int, ...] = (10, 20, 40),
    tail_fractions: tuple[float, ...] = (0.10, 0.15, 0.20),
    book_mode: str = "fixed_count",
    tail_fraction: float | None = None,
    default_borrow_fee_annual_bps: float = 30.0,
    borrow_profile: pd.DataFrame | None = None,
    unavailable_tickers: tuple[str, ...] = (),
) -> dict[str, pd.DataFrame]:
    normalized_forecasts = _normalize_time_columns(forecasts, ["signal_session", "prediction_session"])

    baseline_summary, baseline_daily, baseline_book, baseline_rank_ic = _build_score_summary(
        forecasts=normalized_forecasts,
        score_column=execution_score_column,
        n_long=n_long,
        n_short=n_short,
        costs_bps=costs_bps,
        book_mode=book_mode,
        tail_fraction=tail_fraction,
    )
    baseline_realized = _build_realized_returns(normalized_forecasts)
    baseline_metrics = BacktestRunner().run(baseline_book, baseline_realized, costs_bps=costs_bps)
    baseline_position_returns = baseline_metrics.position_returns
    basket_book_comparison = _build_basket_book_comparison(
        book_frame=baseline_book,
        realized=baseline_realized,
        costs_bps=costs_bps,
    )
    side_book_comparison = _build_side_book_comparison(
        book_frame=baseline_book,
        realized=baseline_realized,
        costs_bps=costs_bps,
    )
    basket_side_summary = _build_basket_side_summary(baseline_position_returns)

    score_behavior = _build_score_behavior_summary(
        normalized_forecasts,
        score_columns=score_behavior_columns,
    )
    signal_strength = _build_signal_strength_summary(
        normalized_forecasts,
        score_column=execution_score_column,
    )
    tail_filter_comparison, tail_filter_daily = _build_tail_filter_comparison(
        normalized_forecasts,
        score_column=execution_score_column,
        tail_fractions=tail_fractions,
        n_short=n_short,
        costs_bps=costs_bps,
    )
    execution_sensitivity, execution_sensitivity_daily = _build_execution_sensitivity(
        normalized_forecasts,
        book_frame=baseline_book,
        costs_bps=costs_bps,
    )
    basket_contribution_summary, basket_contribution_daily = _build_basket_contribution_reports(baseline_position_returns)
    ticker_attribution_summary, ticker_attribution_daily = _build_ticker_attribution_reports(baseline_position_returns)
    live_scoreboard_positions = _build_live_scoreboard_positions(
        baseline_book=baseline_book,
        baseline_position_returns=baseline_position_returns,
        forecasts=normalized_forecasts,
        labeled_frame=labeled_frame,
    )
    live_scoreboard_daily = _build_live_scoreboard_daily(live_scoreboard_positions)
    oos_window_summary, oos_stability_daily, oos_stability_weekly, oos_concentration_summary = _build_oos_window_reports(
        forecasts=normalized_forecasts,
        labeled_frame=labeled_frame,
        live_scoreboard_daily=live_scoreboard_daily,
        baseline_rank_ic=baseline_rank_ic,
    )
    side_isolation_summary, side_isolation_daily, side_isolation_ticker_summary = _build_side_isolation_reports(
        baseline_position_returns
    )
    (
        borrow_overlay_assumptions,
        borrow_overlay_summary,
        borrow_overlay_daily,
        borrow_overlay_positions,
        borrow_overlay_ticker_summary,
    ) = _build_borrow_overlay_reports(
        position_returns=baseline_position_returns,
        costs_bps=costs_bps,
        default_borrow_fee_annual_bps=default_borrow_fee_annual_bps,
        borrow_profile=borrow_profile,
        unavailable_tickers=unavailable_tickers,
    )

    outputs = {
        "signal_execution_baseline_summary": baseline_summary,
        "signal_execution_baseline_daily": baseline_daily,
        "signal_execution_baseline_rank_ic": baseline_rank_ic,
        "signal_execution_baseline_book": baseline_book,
        "score_behavior_summary": score_behavior,
        "signal_strength_summary": signal_strength,
        "tail_filter_comparison": tail_filter_comparison,
        "tail_filter_daily": tail_filter_daily,
        "execution_sensitivity_comparison": execution_sensitivity,
        "execution_sensitivity_daily": execution_sensitivity_daily,
        "basket_book_comparison": basket_book_comparison,
        "side_book_comparison": side_book_comparison,
        "basket_side_summary": basket_side_summary,
        "basket_contribution_summary": basket_contribution_summary,
        "basket_contribution_daily": basket_contribution_daily,
        "ticker_attribution_summary": ticker_attribution_summary,
        "ticker_attribution_daily": ticker_attribution_daily,
        "live_scoreboard_positions": live_scoreboard_positions,
        "live_scoreboard_daily": live_scoreboard_daily,
        "oos_window_summary": oos_window_summary,
        "oos_stability_daily": oos_stability_daily,
        "oos_stability_weekly": oos_stability_weekly,
        "oos_concentration_summary": oos_concentration_summary,
        "side_isolation_summary": side_isolation_summary,
        "side_isolation_daily": side_isolation_daily,
        "side_isolation_ticker_summary": side_isolation_ticker_summary,
        "borrow_overlay_assumptions": borrow_overlay_assumptions,
        "borrow_overlay_summary": borrow_overlay_summary,
        "borrow_overlay_daily": borrow_overlay_daily,
        "borrow_overlay_positions": borrow_overlay_positions,
        "borrow_overlay_ticker_summary": borrow_overlay_ticker_summary,
    }
    return outputs


def write_diagnostic_outputs(output_dir: Path, outputs: dict[str, pd.DataFrame]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for name, frame in outputs.items():
        frame.to_csv(output_dir / f"{name}.csv", index=False)
