from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass
class BaselineScorer:
    benchmark_ticker: str = "SPY"

    def build_realized_daily_returns(self, frame: pd.DataFrame) -> pd.DataFrame:
        candidate = frame[frame["ticker"] != self.benchmark_ticker].copy()
        realized = (
            candidate.groupby(["ticker", "session_date"], as_index=False)
            .agg(realized_day_return=("y_slot", "sum"), realized_future_path=("y_slot", lambda series: series.tolist()))
            .rename(columns={"session_date": "prediction_session"})
            .sort_values(["prediction_session", "ticker"])
            .reset_index(drop=True)
        )
        return realized

    def score(self, frame: pd.DataFrame) -> pd.DataFrame:
        candidate = frame[frame["ticker"] != self.benchmark_ticker].copy()
        session_paths = (
            candidate.pivot_table(index=["ticker", "session_date"], columns="slot_id", values="y_slot", aggfunc="last")
            .sort_index()
            .reset_index()
        )
        session_paths.columns = ["ticker", "session_date"] + [f"slot_{idx}" for idx in range(1, 8)]
        session_paths = session_paths.sort_values(["ticker", "session_date"]).reset_index(drop=True)

        baseline_frames: list[pd.DataFrame] = []
        for _, ticker_frame in session_paths.groupby("ticker", sort=True):
            ordered = ticker_frame.sort_values("session_date").reset_index(drop=True)
            next_session = ordered["session_date"].shift(-1)
            slot_columns = [f"slot_{idx}" for idx in range(1, 8)]

            last_same_slot = ordered.copy()
            last_same_slot["signal_session"] = ordered["session_date"]
            last_same_slot["prediction_session"] = next_session
            last_same_slot["baseline_name"] = "last_same_slot_residual_return"
            last_same_slot["score_q50"] = ordered[slot_columns].sum(axis=1)
            baseline_frames.append(last_same_slot)

            rolling = ordered.copy()
            for column in slot_columns:
                rolling[column] = ordered[column].rolling(window=5, min_periods=5).mean()
            rolling["signal_session"] = ordered["session_date"]
            rolling["prediction_session"] = next_session
            rolling["baseline_name"] = "rolling_5_session_mean_residual_return"
            rolling["score_q50"] = rolling[slot_columns].sum(axis=1)
            baseline_frames.append(rolling)

        baselines = pd.concat(baseline_frames, ignore_index=True)
        baselines = baselines.dropna(subset=["prediction_session"]).copy()
        baselines = baselines[["ticker", "signal_session", "prediction_session", "baseline_name", "score_q50"]]
        baselines["q10_path"] = None
        baselines["q50_path"] = None
        baselines["q90_path"] = None
        return baselines.sort_values(["baseline_name", "signal_session", "ticker"]).reset_index(drop=True)
