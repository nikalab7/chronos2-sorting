from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Iterable
from zoneinfo import ZoneInfo

import pandas as pd
import yfinance as yf


def _normalize_columns(columns: Iterable[object]) -> list[str]:
    normalized: list[str] = []
    for column in columns:
        text = str(column).strip().lower().replace(" ", "_")
        normalized.append(text)
    return normalized


@dataclass
class YahooIntradayProvider:
    def fetch(
        self,
        tickers: list[str],
        start: str | datetime,
        end: str | datetime,
        interval: str = "60m",
        regular_only: bool = True,
        auto_adjust: bool = True,
    ) -> pd.DataFrame:
        unique_tickers = list(dict.fromkeys(tickers))
        raw = yf.download(
            tickers=unique_tickers,
            start=start,
            end=end,
            interval=interval,
            auto_adjust=auto_adjust,
            actions=True,
            prepost=not regular_only,
            progress=False,
            group_by="ticker",
            threads=True,
        )
        if raw.empty:
            return pd.DataFrame(columns=["ticker", "timestamp", "open", "high", "low", "close", "volume"])

        frames: list[pd.DataFrame] = []
        if isinstance(raw.columns, pd.MultiIndex):
            first_level = raw.columns.get_level_values(0).unique().tolist()
            for ticker in first_level:
                ticker_frame = raw[ticker].copy()
                if ticker_frame.dropna(how="all").empty:
                    continue
                ticker_frame.columns = _normalize_columns(ticker_frame.columns)
                ticker_frame["timestamp"] = ticker_frame.index
                ticker_frame["ticker"] = ticker
                frames.append(ticker_frame.reset_index(drop=True))
        else:
            ticker = unique_tickers[0]
            single = raw.copy()
            single.columns = _normalize_columns(single.columns)
            single["timestamp"] = single.index
            single["ticker"] = ticker
            frames.append(single.reset_index(drop=True))

        combined = pd.concat(frames, ignore_index=True)
        required = ["ticker", "timestamp", "open", "high", "low", "close", "volume"]
        missing = [column for column in required if column not in combined.columns]
        if missing:
            raise ValueError(f"Yahoo download is missing required columns: {missing}")

        combined = combined[required].copy()
        combined["timestamp"] = pd.to_datetime(combined["timestamp"])
        return combined.sort_values(["ticker", "timestamp"]).reset_index(drop=True)


@dataclass
class SessionNormalizer:
    market_timezone: str = "America/New_York"
    slot_schedule: dict[str, int] | None = None

    def __post_init__(self) -> None:
        if self.slot_schedule is None:
            self.slot_schedule = {
                "09:30": 1,
                "10:30": 2,
                "11:30": 3,
                "12:30": 4,
                "13:30": 5,
                "14:30": 6,
                "15:30": 7,
            }

    def _to_market_timezone(self, timestamps: pd.Series) -> pd.Series:
        ts = pd.to_datetime(timestamps)
        if ts.dt.tz is None:
            ts = ts.dt.tz_localize(ZoneInfo(self.market_timezone))
        else:
            ts = ts.dt.tz_convert(ZoneInfo(self.market_timezone))
        return ts

    def normalize(self, raw_bars: pd.DataFrame, drop_incomplete_sessions: bool = True) -> pd.DataFrame:
        if raw_bars.empty:
            return pd.DataFrame(
                columns=["ticker", "session_date", "slot_id", "timestamp", "open", "high", "low", "close", "volume"]
            )

        frame = raw_bars.copy()
        frame["timestamp"] = self._to_market_timezone(frame["timestamp"])
        frame["session_date"] = frame["timestamp"].dt.normalize()
        frame["clock_time"] = frame["timestamp"].dt.strftime("%H:%M")
        frame = frame[frame["clock_time"].isin(self.slot_schedule)].copy()
        frame["slot_id"] = frame["clock_time"].map(self.slot_schedule).astype(int)

        frame = frame.drop_duplicates(subset=["ticker", "timestamp"], keep="last")
        frame = frame.sort_values(["ticker", "session_date", "slot_id"])

        if not drop_incomplete_sessions:
            ordered_columns = ["ticker", "session_date", "slot_id", "timestamp", "open", "high", "low", "close", "volume"]
            return frame[ordered_columns].reset_index(drop=True)

        slot_counts = frame.groupby(["ticker", "session_date"])["slot_id"].nunique().rename("slot_count").reset_index()
        complete_sessions = slot_counts[slot_counts["slot_count"] == len(self.slot_schedule)][["ticker", "session_date"]]
        normalized = frame.merge(complete_sessions, on=["ticker", "session_date"], how="inner")

        ordered_columns = ["ticker", "session_date", "slot_id", "timestamp", "open", "high", "low", "close", "volume"]
        return normalized[ordered_columns].reset_index(drop=True)
