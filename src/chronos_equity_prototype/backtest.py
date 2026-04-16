from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .types import Book, MetricsReport


def _annualized_sharpe(returns: pd.Series) -> float:
    clean = returns.dropna()
    if clean.empty:
        return float("nan")
    std = clean.std(ddof=1)
    if std == 0 or np.isnan(std):
        return float("nan")
    return float(np.sqrt(252) * clean.mean() / std)


@dataclass
class BacktestRunner:
    def run(self, book: Book | pd.DataFrame, realized_returns: pd.DataFrame, costs_bps: tuple[int, ...]) -> MetricsReport:
        book_frame = book.frame.copy() if isinstance(book, Book) else book.copy()
        if book_frame.empty:
            empty = pd.DataFrame()
            return MetricsReport(summary=empty, daily_returns=empty, position_returns=empty)

        required_realized = {"ticker", "prediction_session", "realized_day_return"}
        missing = required_realized.difference(realized_returns.columns)
        if missing:
            raise ValueError(f"Realized returns are missing required columns: {sorted(missing)}")

        merged = book_frame.merge(
            realized_returns[["ticker", "prediction_session", "realized_day_return"]],
            on=["ticker", "prediction_session"],
            how="left",
            validate="many_to_one",
        )
        if merged["realized_day_return"].isna().any():
            raise ValueError("The book contains positions without realized next-session returns.")

        merged["gross_position_return"] = merged["weight"] * merged["realized_day_return"]

        daily = (
            merged.groupby("prediction_session", as_index=False)
            .agg(
                gross_return=("gross_position_return", "sum"),
                gross_exposure=("weight", lambda series: float(series.abs().sum())),
            )
            .sort_values("prediction_session")
            .reset_index(drop=True)
        )

        summary_records: list[dict[str, object]] = []
        gross_series = daily["gross_return"]
        summary_records.append(
            {
                "cost_bps": 0,
                "return_type": "gross",
                "mean_daily_return": float(gross_series.mean()),
                "daily_volatility": float(gross_series.std(ddof=1)),
                "annualized_sharpe": _annualized_sharpe(gross_series),
                "win_rate": float((gross_series > 0).mean()),
            }
        )

        for cost_bps in costs_bps:
            cost_rate = cost_bps / 10000.0
            daily[f"net_return_{cost_bps}bps"] = daily["gross_return"] - (daily["gross_exposure"] * cost_rate)
            net_series = daily[f"net_return_{cost_bps}bps"]
            summary_records.append(
                {
                    "cost_bps": cost_bps,
                    "return_type": "net",
                    "mean_daily_return": float(net_series.mean()),
                    "daily_volatility": float(net_series.std(ddof=1)),
                    "annualized_sharpe": _annualized_sharpe(net_series),
                    "win_rate": float((net_series > 0).mean()),
                }
            )

        summary = pd.DataFrame.from_records(summary_records)
        return MetricsReport(summary=summary, daily_returns=daily, position_returns=merged)
