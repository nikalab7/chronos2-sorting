from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class LabelBuilder:
    realized_vol_window: int = 20

    def build(self, frame: pd.DataFrame, benchmark: str = "SPY") -> pd.DataFrame:
        if frame.empty:
            return frame.copy()

        labeled = frame.copy().sort_values(["ticker", "timestamp"]).reset_index(drop=True)
        labeled["slot_log_return"] = np.log(labeled["close"] / labeled["open"])
        labeled["raw_slot_return"] = labeled["slot_log_return"]

        benchmark_frame = labeled[labeled["ticker"] == benchmark][["session_date", "slot_id", "slot_log_return"]].rename(
            columns={"slot_log_return": "benchmark_slot_return"}
        )
        if benchmark_frame.empty:
            raise ValueError(f"Benchmark ticker {benchmark!r} is required in the normalized frame.")

        labeled = labeled.merge(benchmark_frame, on=["session_date", "slot_id"], how="left", validate="many_to_one")
        if labeled["benchmark_slot_return"].isna().any():
            raise ValueError("Missing benchmark slot returns for one or more stock bars.")

        labeled["y_slot"] = labeled["slot_log_return"] - labeled["benchmark_slot_return"]
        labeled["log_volume"] = np.log(labeled["volume"].clip(lower=1.0))
        labeled["high_low_range"] = ((labeled["high"] - labeled["low"]) / labeled["open"].replace(0, np.nan)).fillna(0.0)
        labeled["weekday"] = labeled["session_date"].dt.weekday.astype(float)
        labeled["month_end_flag"] = labeled["session_date"].dt.is_month_end.astype(float)
        labeled["spy_same_slot_return"] = labeled["benchmark_slot_return"]
        labeled["rolling_slot_vol_20"] = (
            labeled.groupby("ticker")["slot_log_return"].transform(
                lambda series: series.rolling(self.realized_vol_window, min_periods=5).std()
            )
        ).fillna(0.0)

        return labeled.sort_values(["ticker", "session_date", "slot_id"]).reset_index(drop=True)
