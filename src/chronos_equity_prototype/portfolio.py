from __future__ import annotations

from dataclasses import dataclass
from math import ceil

import pandas as pd

from .types import Book, ForecastBatch


@dataclass
class PortfolioBuilder:
    def rank_and_size(
        self,
        forecasts: ForecastBatch | pd.DataFrame,
        n_long: int | None,
        n_short: int | None,
        score_column: str = "score_q50",
        *,
        book_mode: str = "fixed_count",
        tail_fraction: float | None = None,
    ) -> Book:
        frame = forecasts.frame.copy() if isinstance(forecasts, ForecastBatch) else forecasts.copy()
        if frame.empty:
            return Book(frame=pd.DataFrame())
        if score_column not in frame.columns:
            raise ValueError(f"Score column {score_column!r} is not present in the forecast frame.")
        if book_mode not in {"fixed_count", "tail_fraction"}:
            raise ValueError(f"Unsupported book_mode: {book_mode!r}")

        extra_columns = [
            column
            for column in ["basket_id", "score_plain", "score_risk", "score_z", "score_q50", "beta_mode"]
            if column in frame.columns
        ]
        records: list[dict[str, object]] = []
        for signal_session, day_frame in frame.groupby("signal_session", sort=True):
            ordered = day_frame.sort_values([score_column, "ticker"], ascending=[False, True]).reset_index(drop=True)
            if book_mode == "fixed_count":
                if n_long is None or n_short is None:
                    raise ValueError("Fixed-count books require both n_long and n_short.")
                if len(ordered) < n_long + n_short:
                    raise ValueError(
                        f"Not enough tickers to build the book for {signal_session.date()}: "
                        f"need {n_long + n_short}, found {len(ordered)}."
                    )
                tail_size = None
                longs = ordered.head(n_long).copy()
                shorts = ordered.tail(n_short).copy()
                long_weight = 0.5 / n_long
                short_weight = -0.5 / n_short
            else:
                if tail_fraction is None or tail_fraction <= 0 or tail_fraction >= 0.5:
                    raise ValueError("Tail-fraction books require tail_fraction to be between 0 and 0.5.")
                tail_size = max(1, int(ceil(len(ordered) * tail_fraction)))
                if (2 * tail_size) > len(ordered):
                    raise ValueError(
                        f"Tail-fraction book would overlap positions for {signal_session.date()}: "
                        f"tail_size={tail_size}, universe={len(ordered)}."
                    )
                longs = ordered.head(tail_size).copy()
                shorts = ordered.tail(tail_size).copy()
                long_weight = 0.5 / tail_size
                short_weight = -0.5 / tail_size

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
                    "book_mode": book_mode,
                    "tail_fraction": tail_fraction if book_mode == "tail_fraction" else float("nan"),
                    "tail_size": tail_size if book_mode == "tail_fraction" else float("nan"),
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
                    "book_mode": book_mode,
                    "tail_fraction": tail_fraction if book_mode == "tail_fraction" else float("nan"),
                    "tail_size": tail_size if book_mode == "tail_fraction" else float("nan"),
                }
                for column in extra_columns:
                    record[column] = row[column]
                records.append(record)

        return Book(frame=pd.DataFrame.from_records(records).sort_values(["signal_session", "side", "ticker"]).reset_index(drop=True))

    def select_winners(
        self,
        forecasts: ForecastBatch | pd.DataFrame,
        *,
        score_column: str = "winner_score",
        max_fraction: float = 0.10,
    ) -> Book:
        frame = forecasts.frame.copy() if isinstance(forecasts, ForecastBatch) else forecasts.copy()
        empty_columns = [
            "signal_session",
            "prediction_session",
            "ticker",
            "side",
            "rank",
            "weight",
            "ranking_score_column",
            "ranking_score",
            "book_mode",
            "tail_fraction",
            "tail_size",
            "book_active",
        ]
        if frame.empty:
            return Book(frame=pd.DataFrame(columns=empty_columns))
        if score_column not in frame.columns:
            raise ValueError(f"Score column {score_column!r} is not present in the forecast frame.")
        if max_fraction <= 0 or max_fraction >= 1:
            raise ValueError("Winner selection requires max_fraction to be between 0 and 1.")

        extra_columns = [
            column
            for column in [
                "basket_id",
                "score_plain",
                "score_risk",
                "score_z",
                "winner_score",
                "pred_ret_1d",
                "pred_alpha_1d",
                "pred_ret_3d",
                "pred_alpha_3d",
                "confidence_proxy",
                "signal_label",
                "buy_eligible",
                "confirmed_3d",
                "beta_mode",
                "product_mode",
            ]
            if column in frame.columns
        ]
        records: list[dict[str, object]] = []
        group_column = "prediction_session" if "prediction_session" in frame.columns else "signal_session"

        for group_value, day_frame in frame.groupby(group_column, sort=True):
            ordered = day_frame.sort_values([score_column, "ticker"], ascending=[False, True]).reset_index(drop=True)
            eligible = ordered.copy()
            if "buy_eligible" in eligible.columns:
                eligible = eligible[eligible["buy_eligible"] == True]  # noqa: E712
            if "confirmed_3d" in eligible.columns:
                eligible = eligible[eligible["confirmed_3d"] == True]  # noqa: E712
            if eligible.empty:
                continue

            max_picks = max(1, int(ceil(len(ordered) * max_fraction)))
            selected = eligible.head(max_picks).copy().reset_index(drop=True)
            weight = 1.0 / len(selected)

            for rank, (_, row) in enumerate(selected.iterrows(), start=1):
                record = {
                    "signal_session": row["signal_session"],
                    "prediction_session": row["prediction_session"],
                    "ticker": row["ticker"],
                    "side": "long",
                    "rank": rank,
                    "weight": weight,
                    "ranking_score_column": score_column,
                    "ranking_score": row[score_column],
                    "book_mode": "winner_top_fraction",
                    "tail_fraction": max_fraction,
                    "tail_size": len(selected),
                    "book_active": True,
                }
                for column in extra_columns:
                    record[column] = row[column]
                records.append(record)

        winner_frame = pd.DataFrame.from_records(records)
        if winner_frame.empty:
            return Book(frame=pd.DataFrame(columns=empty_columns + extra_columns))
        return Book(frame=winner_frame.sort_values(["prediction_session", "rank", "ticker"]).reset_index(drop=True))
