from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from .types import Book, ForecastBatch


@dataclass
class PortfolioBuilder:
    def rank_and_size(self, forecasts: ForecastBatch | pd.DataFrame, n_long: int, n_short: int) -> Book:
        frame = forecasts.frame.copy() if isinstance(forecasts, ForecastBatch) else forecasts.copy()
        if frame.empty:
            return Book(frame=pd.DataFrame())

        records: list[dict[str, object]] = []
        for signal_session, day_frame in frame.groupby("signal_session", sort=True):
            ordered = day_frame.sort_values(["score_q50", "ticker"], ascending=[False, True]).reset_index(drop=True)
            if len(ordered) < n_long + n_short:
                raise ValueError(
                    f"Not enough tickers to build the book for {signal_session.date()}: "
                    f"need {n_long + n_short}, found {len(ordered)}."
                )

            longs = ordered.head(n_long).copy()
            shorts = ordered.tail(n_short).copy()

            long_weight = 0.5 / n_long
            short_weight = -0.5 / n_short

            for rank, (_, row) in enumerate(longs.iterrows(), start=1):
                records.append(
                    {
                        "signal_session": signal_session,
                        "prediction_session": row["prediction_session"],
                        "ticker": row["ticker"],
                        "side": "long",
                        "rank": rank,
                        "weight": long_weight,
                        "score_q50": row["score_q50"],
                    }
                )
            for rank, (_, row) in enumerate(shorts.iterrows(), start=1):
                records.append(
                    {
                        "signal_session": signal_session,
                        "prediction_session": row["prediction_session"],
                        "ticker": row["ticker"],
                        "side": "short",
                        "rank": rank,
                        "weight": short_weight,
                        "score_q50": row["score_q50"],
                    }
                )

        return Book(frame=pd.DataFrame.from_records(records).sort_values(["signal_session", "side", "ticker"]).reset_index(drop=True))
