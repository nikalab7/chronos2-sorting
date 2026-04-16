from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from pandas.tseries.holiday import (
    AbstractHolidayCalendar,
    GoodFriday,
    Holiday,
    USLaborDay,
    USMartinLutherKingJr,
    USMemorialDay,
    USPresidentsDay,
    USThanksgivingDay,
    nearest_workday,
)
from pandas.tseries.offsets import CustomBusinessDay

from .types import StockTask


class USMarketHolidayCalendar(AbstractHolidayCalendar):
    rules = [
        Holiday("New Years Day", month=1, day=1, observance=nearest_workday),
        USMartinLutherKingJr,
        USPresidentsDay,
        GoodFriday,
        USMemorialDay,
        Holiday("Juneteenth", month=6, day=19, observance=nearest_workday, start_date="2021-06-18"),
        Holiday("Independence Day", month=7, day=4, observance=nearest_workday),
        USLaborDay,
        USThanksgivingDay,
        Holiday("Christmas", month=12, day=25, observance=nearest_workday),
    ]


US_MARKET_BUSINESS_DAY = CustomBusinessDay(calendar=USMarketHolidayCalendar())


@dataclass
class ChronosTaskBuilder:
    benchmark_ticker: str = "SPY"
    past_only_covariates: tuple[str, ...] = ("log_volume", "high_low_range", "realized_vol_20", "spy_same_slot_return")
    known_future_covariates: tuple[str, ...] = ("slot_id", "weekday", "month_end_flag")

    @staticmethod
    def next_trading_session(session_date: pd.Timestamp) -> pd.Timestamp:
        return pd.Timestamp(session_date).normalize() + US_MARKET_BUSINESS_DAY

    def _build_future_covariates(self, prediction_session: pd.Timestamp, prediction_length: int) -> dict[str, np.ndarray]:
        prediction_ts = pd.Timestamp(prediction_session)
        return {
            "slot_id": np.arange(1, prediction_length + 1, dtype=float),
            "weekday": np.full(prediction_length, float(prediction_ts.weekday()), dtype=float),
            "month_end_flag": np.full(prediction_length, float(prediction_ts.is_month_end), dtype=float),
        }

    def _assemble_task(
        self,
        *,
        ticker: str,
        signal_session: pd.Timestamp,
        prediction_session: pd.Timestamp,
        history_tail: pd.DataFrame,
        prediction_length: int,
        use_covariates: bool,
        future: pd.DataFrame | None,
    ) -> StockTask:
        past_covariates: dict[str, np.ndarray] = {}
        future_covariates: dict[str, np.ndarray] = {}

        if use_covariates:
            for name in self.past_only_covariates + self.known_future_covariates:
                past_covariates[name] = history_tail[name].to_numpy(dtype=float)
            if future is not None:
                for name in self.known_future_covariates:
                    future_covariates[name] = future[name].to_numpy(dtype=float)
            else:
                future_covariates = self._build_future_covariates(prediction_session, prediction_length)

        realized_future_path = (
            future["y_slot"].to_numpy(dtype=float)
            if future is not None
            else np.full(prediction_length, np.nan, dtype=float)
        )
        realized_day_return = float(future["y_slot"].sum()) if future is not None else float("nan")

        return StockTask(
            ticker=ticker,
            signal_session=pd.Timestamp(signal_session),
            prediction_session=pd.Timestamp(prediction_session),
            target=history_tail["y_slot"].to_numpy(dtype=float),
            past_covariates=past_covariates,
            future_covariates=future_covariates,
            realized_future_path=realized_future_path,
            realized_day_return=realized_day_return,
        )

    def build_tasks(
        self,
        frame: pd.DataFrame,
        context_length: int,
        prediction_length: int,
        use_covariates: bool,
    ) -> list[StockTask]:
        if frame.empty:
            return []

        tasks: list[StockTask] = []
        candidate_frame = frame[frame["ticker"] != self.benchmark_ticker].copy()

        for ticker, ticker_frame in candidate_frame.groupby("ticker", sort=True):
            ordered = ticker_frame.sort_values(["session_date", "slot_id"]).reset_index(drop=True)
            session_dates = ordered["session_date"].drop_duplicates().tolist()
            for session_index in range(len(session_dates) - 1):
                signal_session = session_dates[session_index]
                prediction_session = session_dates[session_index + 1]
                history = ordered[ordered["session_date"] <= signal_session]
                future = ordered[ordered["session_date"] == prediction_session]

                if len(history) < context_length:
                    continue
                if len(future) != prediction_length:
                    continue

                history_tail = history.tail(context_length)
                task = self._assemble_task(
                    ticker=ticker,
                    signal_session=pd.Timestamp(signal_session),
                    prediction_session=pd.Timestamp(prediction_session),
                    history_tail=history_tail,
                    prediction_length=prediction_length,
                    use_covariates=use_covariates,
                    future=future,
                )
                tasks.append(task)

        return tasks

    def build_latest_tasks(
        self,
        frame: pd.DataFrame,
        context_length: int,
        prediction_length: int,
        use_covariates: bool,
    ) -> list[StockTask]:
        if frame.empty:
            return []

        benchmark_frame = frame[frame["ticker"] == self.benchmark_ticker]
        if benchmark_frame.empty:
            raise ValueError(f"Benchmark ticker {self.benchmark_ticker!r} is required to build next-session tasks.")

        latest_signal_session = benchmark_frame["session_date"].max()
        prediction_session = self.next_trading_session(latest_signal_session)

        tasks: list[StockTask] = []
        candidate_frame = frame[frame["ticker"] != self.benchmark_ticker].copy()
        for ticker, ticker_frame in candidate_frame.groupby("ticker", sort=True):
            ordered = ticker_frame.sort_values(["session_date", "slot_id"]).reset_index(drop=True)
            if ordered["session_date"].max() != latest_signal_session:
                continue

            history = ordered[ordered["session_date"] <= latest_signal_session]
            if len(history) < context_length:
                continue

            history_tail = history.tail(context_length)
            task = self._assemble_task(
                ticker=ticker,
                signal_session=pd.Timestamp(latest_signal_session),
                prediction_session=pd.Timestamp(prediction_session),
                history_tail=history_tail,
                prediction_length=prediction_length,
                use_covariates=use_covariates,
                future=None,
            )
            tasks.append(task)

        return tasks

    @staticmethod
    def group_by_signal_session(tasks: list[StockTask]) -> dict[pd.Timestamp, list[StockTask]]:
        grouped: dict[pd.Timestamp, list[StockTask]] = {}
        for task in tasks:
            grouped.setdefault(task.signal_session, []).append(task)
        return grouped
