from __future__ import annotations

from datetime import date
import html
from pathlib import Path

import numpy as np
import pandas as pd


EPSILON = 1e-8


def normalize_time_columns(frame: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    normalized = frame.copy()
    for column in columns:
        if column in normalized.columns:
            normalized[column] = pd.to_datetime(normalized[column], utc=True, errors="coerce")
    return normalized


def default_chronos_output_dir(base_dir: str = "outputs") -> Path:
    return Path(base_dir) / f"chronos_{date.today().isoformat()}"


def _path_text(values: object) -> str:
    if not isinstance(values, list):
        return ""
    return " | ".join(f"h{idx + 1}:{float(value):.4f}" for idx, value in enumerate(values))


def _slot_value(values: object, slot_index: int) -> float:
    if not isinstance(values, list) or slot_index >= len(values):
        return float("nan")
    return float(values[slot_index])


def build_prediction_board(
    forecasts: pd.DataFrame,
    book: pd.DataFrame,
    prediction_summary: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    normalized_forecasts = normalize_time_columns(forecasts, ["signal_session", "prediction_session"])
    normalized_book = normalize_time_columns(book, ["signal_session", "prediction_session"])
    normalized_summary = normalize_time_columns(prediction_summary, ["signal_session", "prediction_session", "generated_at_utc"])

    book_columns = [
        column
        for column in [
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
        ]
        if column in normalized_book.columns
    ]
    board = normalized_forecasts.merge(
        normalized_book[book_columns].drop_duplicates(subset=["signal_session", "prediction_session", "ticker"]),
        on=["signal_session", "prediction_session", "ticker"],
        how="left",
        validate="one_to_one",
    )
    board["signal_date"] = board["signal_session"].dt.strftime("%Y-%m-%d")
    board["prediction_date"] = board["prediction_session"].dt.strftime("%Y-%m-%d")
    board["in_book"] = board["side"].notna()
    board["selection_status"] = np.where(board["in_book"], board["side"], "not_selected")
    board["predicted_direction"] = np.where(
        board["score_plain"] > 0,
        "positive_alpha",
        np.where(board["score_plain"] < 0, "negative_alpha", "flat"),
    )

    alpha_uncertainty: list[float] = []
    for row in board.itertuples(index=False):
        stock_width = np.array(row.stock_q90_path, dtype=float) - np.array(row.stock_q10_path, dtype=float)
        spy_width = np.array(row.spy_q90_path, dtype=float) - np.array(row.spy_q10_path, dtype=float)
        alpha_uncertainty.append(float(np.sum(stock_width + (abs(float(row.beta_used)) * spy_width))))
    board["alpha_uncertainty"] = alpha_uncertainty
    board["confidence_proxy"] = board["score_plain"].abs() / (EPSILON + board["alpha_uncertainty"])
    board["prediction_rank_desc"] = board.groupby("prediction_session")["score_z"].rank(method="dense", ascending=False)
    board["confidence_rank_desc"] = board.groupby("prediction_session")["confidence_proxy"].rank(method="dense", ascending=False)
    board["edge_rank_desc"] = board.groupby("prediction_session")["score_plain"].transform(
        lambda series: series.abs().rank(method="dense", ascending=False)
    )
    board["alpha_q50_path_text"] = board["alpha_q50_path"].apply(_path_text)
    board["stock_q50_path_text"] = board["stock_q50_path"].apply(_path_text)
    board["spy_q50_path_text"] = board["spy_q50_path"].apply(_path_text)
    for slot_index in range(7):
        slot_id = slot_index + 1
        board[f"alpha_q50_h{slot_id}"] = board["alpha_q50_path"].apply(lambda values: _slot_value(values, slot_index))
        board[f"stock_q50_h{slot_id}"] = board["stock_q50_path"].apply(lambda values: _slot_value(values, slot_index))
        board[f"spy_q50_h{slot_id}"] = board["spy_q50_path"].apply(lambda values: _slot_value(values, slot_index))

    ordered_columns = [
        "signal_date",
        "prediction_date",
        "signal_session",
        "prediction_session",
        "ticker",
        "basket_id",
        "in_book",
        "selection_status",
        "side",
        "rank",
        "prediction_rank_desc",
        "confidence_rank_desc",
        "edge_rank_desc",
        "predicted_direction",
        "score_plain",
        "score_z",
        "score_risk",
        "confidence_proxy",
        "alpha_uncertainty",
        "beta_used",
        "beta_mode",
        "weight",
        "ranking_score_column",
        "ranking_score",
        "book_mode",
        "tail_fraction",
        "tail_size",
        "alpha_q50_path_text",
        "stock_q50_path_text",
        "spy_q50_path_text",
    ] + [
        column
        for column in board.columns
        if column.startswith("alpha_q50_h") or column.startswith("stock_q50_h") or column.startswith("spy_q50_h")
    ]
    for column in ordered_columns:
        if column not in board.columns:
            board[column] = np.nan
    board = board[ordered_columns].sort_values(["prediction_session", "prediction_rank_desc", "ticker"]).reset_index(drop=True)

    slot_records: list[dict[str, object]] = []
    for row in normalized_forecasts.itertuples(index=False):
        alpha_uncertainty_total = float(
            np.sum(
                (np.array(row.stock_q90_path, dtype=float) - np.array(row.stock_q10_path, dtype=float))
                + (abs(float(row.beta_used)) * (np.array(row.spy_q90_path, dtype=float) - np.array(row.spy_q10_path, dtype=float)))
            )
        )
        for slot_index in range(len(row.alpha_q50_path)):
            slot_records.append(
                {
                    "signal_session": row.signal_session,
                    "prediction_session": row.prediction_session,
                    "signal_date": pd.Timestamp(row.signal_session).strftime("%Y-%m-%d"),
                    "prediction_date": pd.Timestamp(row.prediction_session).strftime("%Y-%m-%d"),
                    "ticker": row.ticker,
                    "basket_id": row.basket_id,
                    "slot_id": slot_index + 1,
                    "alpha_q50": float(row.alpha_q50_path[slot_index]),
                    "stock_q10": float(row.stock_q10_path[slot_index]),
                    "stock_q50": float(row.stock_q50_path[slot_index]),
                    "stock_q90": float(row.stock_q90_path[slot_index]),
                    "spy_q10": float(row.spy_q10_path[slot_index]),
                    "spy_q50": float(row.spy_q50_path[slot_index]),
                    "spy_q90": float(row.spy_q90_path[slot_index]),
                    "slot_interval_width": float(
                        (row.stock_q90_path[slot_index] - row.stock_q10_path[slot_index])
                        + (abs(float(row.beta_used)) * (row.spy_q90_path[slot_index] - row.spy_q10_path[slot_index]))
                    ),
                    "total_alpha_uncertainty": alpha_uncertainty_total,
                }
            )
    slot_board = pd.DataFrame.from_records(slot_records).sort_values(
        ["prediction_session", "ticker", "slot_id"]
    ).reset_index(drop=True)

    summary_rows: list[dict[str, object]] = []
    for prediction_session, session_frame in board.groupby("prediction_session", sort=True):
        book_frame = session_frame[session_frame["in_book"]].copy()
        longs = book_frame[book_frame["side"] == "long"].sort_values(["rank", "ticker"])
        shorts = book_frame[book_frame["side"] == "short"].sort_values(["rank", "ticker"])
        summary_rows.append(
            {
                "signal_session_count": int(session_frame["signal_session"].nunique()),
                "prediction_session": prediction_session,
                "prediction_date": pd.Timestamp(prediction_session).strftime("%Y-%m-%d"),
                "n_forecasts": int(len(session_frame)),
                "n_book_positions": int(len(book_frame)),
                "n_longs": int((book_frame["side"] == "long").sum()),
                "n_shorts": int((book_frame["side"] == "short").sum()),
                "top_longs": " | ".join(f"{row.ticker}:{row.score_z:.3f}" for row in longs.itertuples(index=False)),
                "top_shorts": " | ".join(f"{row.ticker}:{row.score_z:.3f}" for row in shorts.itertuples(index=False)),
                "highest_confidence_ticker": (
                    session_frame.sort_values(["confidence_proxy", "ticker"], ascending=[False, True]).iloc[0]["ticker"]
                    if not session_frame.empty
                    else ""
                ),
                "highest_confidence_proxy": float(session_frame["confidence_proxy"].max()) if not session_frame.empty else float("nan"),
                "mixed_signal_sessions": bool(session_frame["signal_session"].nunique() > 1),
            }
        )

    if normalized_summary is not None and not normalized_summary.empty:
        summary_board = pd.DataFrame(summary_rows).merge(
            normalized_summary.drop_duplicates(subset=["prediction_session"])[
                [
                    column
                    for column in [
                        "prediction_session",
                        "signal_session",
                        "generated_at_utc",
                        "model_id",
                        "context_length",
                        "basket_mode",
                        "beta_mode",
                        "book_mode",
                        "tail_fraction",
                        "tail_size",
                        "score_column",
                        "observed_latest_session",
                        "has_newer_observed_session_than_signal",
                        "prediction_freshness",
                        "prediction_freshness_note",
                    ]
                    if column in normalized_summary.columns
                ]
            ],
            on="prediction_session",
            how="left",
        )
    else:
        summary_board = pd.DataFrame(summary_rows)

    warning_records: list[dict[str, object]] = []
    if board["prediction_session"].nunique() > 1:
        warning_records.append(
            {
                "warning_type": "multiple_prediction_sessions",
                "warning_message": "The current prediction package contains more than one prediction_session. Read signal_session and prediction_session explicitly per row.",
                "count": int(board["prediction_session"].nunique()),
            }
        )
    if board["signal_session"].nunique() > 1:
        warning_records.append(
            {
                "warning_type": "multiple_signal_sessions",
                "warning_message": "Different baskets may have rolled from different latest signal sessions. Treat each prediction_session separately.",
                "count": int(board["signal_session"].nunique()),
            }
        )
    warnings = pd.DataFrame.from_records(warning_records)
    return board, slot_board, summary_board


def build_backtest_boards(analysis_dir: Path) -> dict[str, pd.DataFrame]:
    scorecard = pd.read_csv(analysis_dir / "backtest_scorecard.csv")
    daily = pd.read_csv(analysis_dir / "live_scoreboard_daily.csv")
    positions = pd.read_csv(analysis_dir / "live_scoreboard_positions.csv")
    ticker_summary = pd.read_csv(analysis_dir / "ticker_attribution_summary.csv")
    rank_ic = pd.read_csv(analysis_dir / "signal_execution_baseline_rank_ic.csv")

    daily = normalize_time_columns(daily, ["signal_session", "prediction_session"])
    positions = normalize_time_columns(positions, ["signal_session", "prediction_session"])
    rank_ic = normalize_time_columns(rank_ic, ["prediction_session"])

    daily_rank_ic = rank_ic[rank_ic["prediction_session"].notna()][["prediction_session", "rank_ic"]].copy()
    daily_board = daily.merge(daily_rank_ic, on="prediction_session", how="left", validate="one_to_one")
    daily_board["signal_date"] = daily_board["signal_session"].dt.strftime("%Y-%m-%d")
    daily_board["prediction_date"] = daily_board["prediction_session"].dt.strftime("%Y-%m-%d")
    daily_board["day_quality_rank"] = daily_board["gross_long_short_pnl"].rank(method="dense", ascending=False)
    daily_board = daily_board.sort_values("prediction_session").reset_index(drop=True)

    positions["predicted_global_rank"] = positions.groupby("prediction_session")["score_z"].rank(method="dense", ascending=False)
    positions["strength_rank"] = positions.groupby("prediction_session")["score_plain"].transform(
        lambda series: series.abs().rank(method="dense", ascending=False)
    )
    positions["realized_contribution_rank"] = positions.groupby("prediction_session")["gross_position_return"].rank(method="dense", ascending=False)
    positions["alpha_gap"] = positions["next_day_realized_alpha"] - positions["score_plain"]
    positions["sign_hit"] = np.sign(positions["score_plain"]) == np.sign(positions["next_day_realized_alpha"])
    positions["prediction_date"] = positions["prediction_session"].dt.strftime("%Y-%m-%d")
    positions["signal_date"] = positions["signal_session"].dt.strftime("%Y-%m-%d")
    position_board = positions.sort_values(["prediction_session", "predicted_global_rank", "ticker"]).reset_index(drop=True)

    ticker_board = ticker_summary.copy()
    ticker_board["contributor_rank"] = ticker_board["total_gross_pnl"].rank(method="dense", ascending=False)
    ticker_board["drag_rank"] = ticker_board["total_gross_pnl"].rank(method="dense", ascending=True)
    ticker_board = ticker_board.sort_values(["contributor_rank", "ticker"]).reset_index(drop=True)

    return {
        "backtest_overview": scorecard,
        "backtest_daily_board": daily_board,
        "backtest_position_board": position_board,
        "backtest_ticker_board": ticker_board,
    }


def build_live_report_boards(
    ledger_dir: Path,
    *,
    prediction_session: pd.Timestamp | None,
) -> dict[str, pd.DataFrame]:
    if not ledger_dir.exists():
        return {}

    files = {
        "live_scoreboard_positions": "live_scoreboard_positions.csv",
        "live_scoreboard_daily": "live_scoreboard_daily.csv",
        "live_winner_positions": "live_winner_positions.csv",
        "live_winner_daily_summary": "live_winner_daily_summary.csv",
        "live_winner_backtest_summary": "live_winner_backtest_summary.csv",
        "live_winner_rank_ic": "live_winner_rank_ic.csv",
        "live_execution_position_returns": "live_execution_position_returns.csv",
        "live_execution_daily_returns": "live_execution_daily_returns.csv",
        "live_metrics_summary": "live_metrics_summary.csv",
        "live_execution_sensitivity_comparison": "live_execution_sensitivity_comparison.csv",
        "reconciliation_summary": "reconciliation_summary.csv",
        "prediction_summary_history": "prediction_summary_history.csv",
    }
    frames: dict[str, pd.DataFrame] = {}
    for name, filename in files.items():
        path = ledger_dir / filename
        if path.exists():
            frames[name] = pd.read_csv(path)

    for name in [
        "live_scoreboard_positions",
        "live_scoreboard_daily",
        "live_winner_positions",
        "live_winner_daily_summary",
        "live_execution_position_returns",
        "live_execution_daily_returns",
        "prediction_summary_history",
    ]:
        if name in frames:
            frames[name] = normalize_time_columns(
                frames[name],
                ["signal_session", "prediction_session", "generated_at_utc"],
            )

    if prediction_session is None:
        return frames

    if "live_scoreboard_positions" in frames:
        frames["live_current_day_positions"] = (
            frames["live_scoreboard_positions"]
            .loc[frames["live_scoreboard_positions"]["prediction_session"] == prediction_session]
            .sort_values(["side", "rank", "ticker"])
            .reset_index(drop=True)
        )
    if "live_scoreboard_daily" in frames:
        frames["live_current_day_daily"] = (
            frames["live_scoreboard_daily"]
            .loc[frames["live_scoreboard_daily"]["prediction_session"] == prediction_session]
            .sort_values(["prediction_session"])
            .reset_index(drop=True)
        )
    if "live_execution_position_returns" in frames:
        frames["live_execution_current_day_positions"] = (
            frames["live_execution_position_returns"]
            .loc[frames["live_execution_position_returns"]["prediction_session"] == prediction_session]
            .sort_values(["execution_mode", "side", "rank", "ticker"])
            .reset_index(drop=True)
        )
    if "live_execution_daily_returns" in frames:
        frames["live_execution_current_day_daily"] = (
            frames["live_execution_daily_returns"]
            .loc[frames["live_execution_daily_returns"]["prediction_session"] == prediction_session]
            .sort_values(["execution_mode", "prediction_session"])
            .reset_index(drop=True)
        )
    if "prediction_summary_history" in frames:
        frames["prediction_summary_current"] = (
            frames["prediction_summary_history"]
            .loc[frames["prediction_summary_history"]["prediction_session"] == prediction_session]
            .sort_values(["prediction_session"])
            .reset_index(drop=True)
        )
    if "live_winner_positions" in frames:
        frames["live_winner_positions_current"] = (
            frames["live_winner_positions"]
            .loc[frames["live_winner_positions"]["prediction_session"] == prediction_session]
            .sort_values(["rank", "ticker"])
            .reset_index(drop=True)
        )
    if "live_winner_daily_summary" in frames:
        frames["live_winner_daily_summary_current"] = (
            frames["live_winner_daily_summary"]
            .loc[frames["live_winner_daily_summary"]["prediction_session"] == prediction_session]
            .sort_values(["prediction_session"])
            .reset_index(drop=True)
        )
    return frames


def _float_or_nan(value: object) -> float:
    if value is None:
        return float("nan")
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def _summary_metric(
    summary_frame: pd.DataFrame,
    *,
    cost_bps: int,
    column: str = "annualized_sharpe",
) -> float:
    if summary_frame.empty or "cost_bps" not in summary_frame.columns or column not in summary_frame.columns:
        return float("nan")
    match = summary_frame.loc[summary_frame["cost_bps"] == cost_bps]
    if match.empty:
        return float("nan")
    return _float_or_nan(match.iloc[0][column])


def _detect_product_mode(
    *,
    prediction_summary: pd.DataFrame | None = None,
    forecasts: pd.DataFrame | None = None,
) -> str:
    for frame in [prediction_summary, forecasts]:
        if frame is None or frame.empty or "product_mode" not in frame.columns:
            continue
        values = frame["product_mode"].dropna()
        if not values.empty:
            return str(values.iloc[0])
    return "ranker"


def _build_winner_paper_summary(
    *,
    prediction_summary: pd.DataFrame,
    winner_board: pd.DataFrame,
    winner_top_picks: pd.DataFrame,
    winner_daily_summary: pd.DataFrame,
    live_boards: dict[str, pd.DataFrame],
    backtest_boards: dict[str, pd.DataFrame],
    report_mode: str,
    report_context: dict[str, object] | None,
) -> pd.DataFrame:
    report_context = report_context or {}
    summary_row = prediction_summary.iloc[0].to_dict() if not prediction_summary.empty else {}
    daily_row = winner_daily_summary.iloc[0].to_dict() if not winner_daily_summary.empty else {}
    paper_summary: dict[str, object] = {
        "report_generated_utc": pd.Timestamp.utcnow(),
        "report_mode": report_mode,
        "product_mode": "winner_picker",
        "prediction_dir": str(report_context.get("prediction_dir", "")),
        "analysis_dir": str(report_context.get("analysis_dir", "")),
        "ledger_dir": str(report_context.get("ledger_dir", "")),
        "reconciliation_status": report_context.get("reconciliation_status", "unknown"),
        "reconciliation_message": report_context.get("reconciliation_message", ""),
        **summary_row,
        **daily_row,
    }
    if not winner_board.empty:
        paper_summary["n_forecasts"] = int(len(winner_board))
        paper_summary["winner_board_top_ticker"] = str(
            winner_board.sort_values(["winner_score", "ticker"], ascending=[False, True]).iloc[0]["ticker"]
        )
        paper_summary["winner_board_top_confidence_ticker"] = str(
            winner_board.sort_values(["confidence_proxy", "ticker"], ascending=[False, True]).iloc[0]["ticker"]
        )
    if not winner_top_picks.empty:
        paper_summary["n_selected"] = int(len(winner_top_picks))
        paper_summary["selected_tickers"] = " | ".join(winner_top_picks.sort_values(["rank", "ticker"])["ticker"].astype(str).tolist())
    live_current = live_boards.get("live_winner_daily_summary_current", pd.DataFrame())
    if not live_current.empty:
        live_row = live_current.iloc[0]
        paper_summary["live_current_available"] = True
        paper_summary["live_current_gross_return"] = _float_or_nan(live_row.get("gross_return"))
        paper_summary["live_current_mean_selected_return"] = _float_or_nan(live_row.get("mean_selected_return"))
        paper_summary["live_current_mean_selected_alpha"] = _float_or_nan(live_row.get("mean_selected_alpha"))
    else:
        paper_summary["live_current_available"] = False
    live_summary = live_boards.get("live_winner_backtest_summary", pd.DataFrame())
    if not live_summary.empty:
        for column in live_summary.columns:
            paper_summary[f"live_{column}"] = live_summary.iloc[0][column]
    backtest_summary = backtest_boards.get("winner_backtest_summary", pd.DataFrame())
    if not backtest_summary.empty:
        for column in backtest_summary.columns:
            paper_summary[f"backtest_{column}"] = backtest_summary.iloc[0][column]
    return pd.DataFrame([paper_summary])


def _build_paper_summary(
    *,
    prediction_summary_board: pd.DataFrame,
    prediction_board: pd.DataFrame,
    live_boards: dict[str, pd.DataFrame],
    backtest_boards: dict[str, pd.DataFrame],
    report_mode: str,
    report_context: dict[str, object] | None,
) -> pd.DataFrame:
    report_context = report_context or {}
    summary_row = prediction_summary_board.iloc[0].to_dict() if not prediction_summary_board.empty else {}
    paper_summary: dict[str, object] = {
        "report_generated_utc": pd.Timestamp.utcnow(),
        "report_mode": report_mode,
        "prediction_dir": str(report_context.get("prediction_dir", "")),
        "analysis_dir": str(report_context.get("analysis_dir", "")),
        "ledger_dir": str(report_context.get("ledger_dir", "")),
        "reconciliation_status": report_context.get("reconciliation_status", "unknown"),
        "reconciliation_message": report_context.get("reconciliation_message", ""),
        **summary_row,
    }

    if not prediction_board.empty:
        paper_summary["book_selected_count"] = int(prediction_board["in_book"].sum())
        paper_summary["not_selected_count"] = int((~prediction_board["in_book"]).sum())
        paper_summary["universe_top_score_ticker"] = str(
            prediction_board.sort_values(["prediction_rank_desc", "ticker"]).iloc[0]["ticker"]
        )
        paper_summary["universe_top_confidence_ticker"] = str(
            prediction_board.sort_values(["confidence_rank_desc", "ticker"]).iloc[0]["ticker"]
        )

    live_current_day = live_boards.get("live_current_day_daily", pd.DataFrame())
    if not live_current_day.empty:
        live_row = live_current_day.iloc[0]
        paper_summary["live_current_day_available"] = True
        paper_summary["live_current_long_short_spread"] = _float_or_nan(live_row.get("long_short_spread"))
        paper_summary["live_current_gross_long_short_pnl"] = _float_or_nan(live_row.get("gross_long_short_pnl"))
        paper_summary["live_current_best_contributor"] = live_row.get("best_contributor", "")
        paper_summary["live_current_worst_contributor"] = live_row.get("worst_contributor", "")
    else:
        paper_summary["live_current_day_available"] = False

    live_history = live_boards.get("live_scoreboard_daily", pd.DataFrame())
    if not live_history.empty:
        paper_summary["live_history_days"] = int(live_history["prediction_session"].nunique())
        paper_summary["live_history_book_hit_rate"] = float(live_history["book_hit"].mean()) if "book_hit" in live_history.columns else float("nan")

    live_metrics = live_boards.get("live_metrics_summary", pd.DataFrame())
    if not live_metrics.empty:
        paper_summary["live_gross_sharpe"] = _summary_metric(live_metrics, cost_bps=0)
        paper_summary["live_net_sharpe_10bps"] = _summary_metric(live_metrics, cost_bps=10)
        paper_summary["live_net_sharpe_20bps"] = _summary_metric(live_metrics, cost_bps=20)
        paper_summary["live_net_sharpe_40bps"] = _summary_metric(live_metrics, cost_bps=40)

    backtest_overview = backtest_boards.get("backtest_overview", pd.DataFrame())
    if not backtest_overview.empty:
        backtest_row = backtest_overview.iloc[0]
        for column in [
            "universe_size",
            "prediction_days",
            "mean_rank_ic",
            "positive_rank_ic_rate",
            "gross_sharpe",
            "net_sharpe_10bps",
            "net_sharpe_20bps",
            "net_sharpe_40bps",
            "long_hit_rate",
            "short_hit_rate",
            "book_hit_rate",
            "top1_capture_mean",
            "top5_capture_mean",
            "mean_daily_long_short_spread",
            "top_contributors",
            "top_drags",
        ]:
            if column in backtest_row:
                paper_summary[f"backtest_{column}"] = backtest_row[column]

    return pd.DataFrame([paper_summary])


def _html_table(
    frame: pd.DataFrame,
    *,
    columns: list[str] | None = None,
    max_rows: int | None = None,
) -> str:
    if frame.empty:
        return "<p class='muted'>No rows available.</p>"
    table = frame.copy()
    if columns is not None:
        available = [column for column in columns if column in table.columns]
        table = table[available]
    if max_rows is not None:
        table = table.head(max_rows)
    return table.to_html(index=False, border=0, classes="dataframe chronos-table")


def _write_bar_chart(
    path: Path,
    *,
    labels: list[str],
    values: list[float],
    colors: list[str],
    title: str,
    ylabel: str,
    horizontal: bool = False,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(10, 5))
    positions = np.arange(len(labels))
    if horizontal:
        ax.barh(positions, values, color=colors, alpha=0.85)
        ax.set_yticks(positions)
        ax.set_yticklabels(labels)
        ax.axvline(0.0, color="#555555", linewidth=1.0)
        ax.set_xlabel(ylabel)
    else:
        ax.bar(positions, values, color=colors, alpha=0.85)
        ax.set_xticks(positions)
        ax.set_xticklabels(labels, rotation=45, ha="right")
        ax.axhline(0.0, color="#555555", linewidth=1.0)
        ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, axis="y" if not horizontal else "x", alpha=0.2)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _write_line_chart(
    path: Path,
    *,
    x_values: list[float],
    series: list[tuple[str, list[float], str]],
    title: str,
    ylabel: str,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(10, 5))
    for label, values, color in series:
        ax.plot(x_values, values, marker="o", linewidth=2.0, label=label, color=color)
    ax.set_title(title)
    ax.set_xlabel("Slot")
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.2)
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _write_scatter_chart(
    path: Path,
    *,
    x_values: list[float],
    y_values: list[float],
    labels: list[str],
    colors: list[str],
    title: str,
    xlabel: str,
    ylabel: str,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(x_values, y_values, c=colors, alpha=0.85)
    for x_value, y_value, label in zip(x_values, y_values, labels):
        ax.annotate(label, (x_value, y_value), fontsize=8, xytext=(4, 4), textcoords="offset points")
    ax.axhline(0.0, color="#555555", linewidth=1.0)
    ax.axvline(0.0, color="#555555", linewidth=1.0)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.2)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _write_report_charts(
    output_dir: Path,
    *,
    prediction_board: pd.DataFrame,
    live_boards: dict[str, pd.DataFrame],
    backtest_boards: dict[str, pd.DataFrame],
) -> pd.DataFrame:
    charts_dir = output_dir / "charts"
    charts_dir.mkdir(parents=True, exist_ok=True)

    chart_rows: list[dict[str, str]] = []
    current_book = prediction_board[prediction_board["in_book"]].copy()
    if not current_book.empty:
        ordered_book = current_book.sort_values(["side", "rank", "ticker"]).reset_index(drop=True)
        score_chart = charts_dir / "current_book_score_z.png"
        _write_bar_chart(
            score_chart,
            labels=ordered_book["ticker"].tolist(),
            values=ordered_book["score_z"].astype(float).tolist(),
            colors=["#2e8b57" if side == "long" else "#b22222" for side in ordered_book["side"]],
            title="Current Book Ranking Score (score_z)",
            ylabel="score_z",
        )
        chart_rows.append(
            {
                "chart_key": "current_book_score_z",
                "title": "Current Book Ranking Score",
                "description": "Tail-selected longs and shorts ranked by score_z.",
                "relative_path": str(score_chart.relative_to(output_dir)).replace("\\", "/"),
            }
        )

        confidence_chart = charts_dir / "current_book_confidence.png"
        _write_bar_chart(
            confidence_chart,
            labels=ordered_book["ticker"].tolist(),
            values=ordered_book["confidence_proxy"].astype(float).tolist(),
            colors=["#1f77b4"] * len(ordered_book),
            title="Current Book Confidence",
            ylabel="confidence_proxy",
        )
        chart_rows.append(
            {
                "chart_key": "current_book_confidence",
                "title": "Current Book Confidence",
                "description": "abs(score_plain) divided by alpha uncertainty for each selected ticker.",
                "relative_path": str(confidence_chart.relative_to(output_dir)).replace("\\", "/"),
            }
        )

        path_chart = charts_dir / "current_book_alpha_paths.png"
        _write_line_chart(
            path_chart,
            x_values=list(range(1, 8)),
            series=[
                (
                    f"{row.ticker} ({row.side})",
                    [float(getattr(row, f"alpha_q50_h{slot_id}")) for slot_id in range(1, 8)],
                    "#2e8b57" if row.side == "long" else "#b22222",
                )
                for row in ordered_book.itertuples(index=False)
            ],
            title="Predicted Alpha Path Across The Next Session",
            ylabel="alpha q50",
        )
        chart_rows.append(
            {
                "chart_key": "current_book_alpha_paths",
                "title": "Current Book Alpha Paths",
                "description": "Seven-slot predicted alpha path for the selected names.",
                "relative_path": str(path_chart.relative_to(output_dir)).replace("\\", "/"),
            }
        )

    if not prediction_board.empty:
        universe_chart = charts_dir / "prediction_universe_edge_vs_confidence.png"
        ordered_universe = prediction_board.sort_values(["prediction_rank_desc", "ticker"]).reset_index(drop=True)
        _write_scatter_chart(
            universe_chart,
            x_values=ordered_universe["score_plain"].astype(float).tolist(),
            y_values=ordered_universe["confidence_proxy"].astype(float).tolist(),
            labels=ordered_universe["ticker"].astype(str).tolist(),
            colors=["#ff8c00" if selected else "#4f81bd" for selected in ordered_universe["in_book"]],
            title="Predicted Edge Versus Confidence",
            xlabel="score_plain",
            ylabel="confidence_proxy",
        )
        chart_rows.append(
            {
                "chart_key": "prediction_universe_edge_vs_confidence",
                "title": "Prediction Universe Edge Versus Confidence",
                "description": "Orange names are in the tail book. Blue names were forecasted but not selected.",
                "relative_path": str(universe_chart.relative_to(output_dir)).replace("\\", "/"),
            }
        )

    live_current_positions = live_boards.get("live_current_day_positions", pd.DataFrame())
    if not live_current_positions.empty:
        realized_chart = charts_dir / "current_day_predicted_vs_realized.png"
        ordered_live = live_current_positions.sort_values(["side", "rank", "ticker"]).reset_index(drop=True)
        _write_scatter_chart(
            realized_chart,
            x_values=ordered_live["score_plain"].astype(float).tolist(),
            y_values=ordered_live["next_day_realized_alpha"].astype(float).tolist(),
            labels=ordered_live["ticker"].astype(str).tolist(),
            colors=["#2e8b57" if side == "long" else "#b22222" for side in ordered_live["side"]],
            title="Current Day Predicted Edge Versus Realized Alpha",
            xlabel="predicted score_plain",
            ylabel="realized alpha",
        )
        chart_rows.append(
            {
                "chart_key": "current_day_predicted_vs_realized",
                "title": "Current Day Predicted Versus Realized",
                "description": "Only available after reconciliation once the predicted session has finished.",
                "relative_path": str(realized_chart.relative_to(output_dir)).replace("\\", "/"),
            }
        )

    live_daily = live_boards.get("live_scoreboard_daily", pd.DataFrame())
    if not live_daily.empty and "gross_long_short_pnl" in live_daily.columns:
        cumulative_chart = charts_dir / "live_cumulative_gross_pnl.png"
        ordered_live_daily = live_daily.sort_values("prediction_session").reset_index(drop=True).copy()
        ordered_live_daily["cumulative_gross_pnl"] = ordered_live_daily["gross_long_short_pnl"].astype(float).cumsum()
        _write_line_chart(
            cumulative_chart,
            x_values=list(range(1, len(ordered_live_daily) + 1)),
            series=[("live cumulative", ordered_live_daily["cumulative_gross_pnl"].tolist(), "#1f77b4")],
            title="Live Scoreboard Cumulative Gross PnL",
            ylabel="cumulative gross pnl",
        )
        chart_rows.append(
            {
                "chart_key": "live_cumulative_gross_pnl",
                "title": "Live Cumulative Gross PnL",
                "description": "Cumulative gross long-short PnL across reconciled live prediction days.",
                "relative_path": str(cumulative_chart.relative_to(output_dir)).replace("\\", "/"),
            }
        )

    backtest_daily = backtest_boards.get("backtest_daily_board", pd.DataFrame())
    if not backtest_daily.empty and "gross_long_short_pnl" in backtest_daily.columns:
        backtest_chart = charts_dir / "backtest_daily_gross_pnl.png"
        ordered_backtest = backtest_daily.sort_values("prediction_session").reset_index(drop=True)
        _write_line_chart(
            backtest_chart,
            x_values=list(range(1, len(ordered_backtest) + 1)),
            series=[("backtest daily", ordered_backtest["gross_long_short_pnl"].astype(float).tolist(), "#6a5acd")],
            title="Historical Backtest Daily Gross PnL",
            ylabel="gross long-short pnl",
        )
        chart_rows.append(
            {
                "chart_key": "backtest_daily_gross_pnl",
                "title": "Historical Backtest Daily Gross PnL",
                "description": "Daily gross long-short PnL from the frozen backtest analysis.",
                "relative_path": str(backtest_chart.relative_to(output_dir)).replace("\\", "/"),
            }
        )

    backtest_tickers = backtest_boards.get("backtest_ticker_board", pd.DataFrame())
    if not backtest_tickers.empty and "total_gross_pnl" in backtest_tickers.columns:
        ticker_chart = charts_dir / "backtest_top_and_bottom_tickers.png"
        top = backtest_tickers.sort_values(["total_gross_pnl", "ticker"], ascending=[False, True]).head(5)
        bottom = backtest_tickers.sort_values(["total_gross_pnl", "ticker"], ascending=[True, True]).head(5)
        combined = pd.concat([top, bottom], ignore_index=True)
        _write_bar_chart(
            ticker_chart,
            labels=combined["ticker"].astype(str).tolist(),
            values=combined["total_gross_pnl"].astype(float).tolist(),
            colors=["#2e8b57" if value >= 0 else "#b22222" for value in combined["total_gross_pnl"].astype(float)],
            title="Historical Biggest Contributors And Drags",
            ylabel="total gross pnl",
            horizontal=True,
        )
        chart_rows.append(
            {
                "chart_key": "backtest_top_and_bottom_tickers",
                "title": "Historical Contributors And Drags",
                "description": "Top and bottom historical ticker contributors from the frozen backtest.",
                "relative_path": str(ticker_chart.relative_to(output_dir)).replace("\\", "/"),
            }
        )

    charts = pd.DataFrame.from_records(chart_rows)
    if not charts.empty:
        charts.to_csv(output_dir / "chart_manifest.csv", index=False)
    return charts


def _write_winner_report_charts(
    output_dir: Path,
    *,
    winner_board: pd.DataFrame,
    winner_top_picks: pd.DataFrame,
    live_boards: dict[str, pd.DataFrame],
) -> pd.DataFrame:
    charts_dir = output_dir / "charts"
    charts_dir.mkdir(parents=True, exist_ok=True)

    chart_rows: list[dict[str, str]] = []
    if not winner_top_picks.empty:
        ordered_picks = winner_top_picks.sort_values(["rank", "ticker"]).reset_index(drop=True)
        score_chart = charts_dir / "winner_top_picks_score.png"
        _write_bar_chart(
            score_chart,
            labels=ordered_picks["ticker"].astype(str).tolist(),
            values=ordered_picks["winner_score"].astype(float).tolist(),
            colors=["#2e8b57"] * len(ordered_picks),
            title="Winner Top Picks",
            ylabel="winner_score",
        )
        chart_rows.append(
            {
                "chart_key": "winner_top_picks_score",
                "title": "Winner Top Picks",
                "description": "Selected winner-mode longs ranked by winner_score.",
                "relative_path": str(score_chart.relative_to(output_dir)).replace("\\", "/"),
            }
        )

    if not winner_board.empty:
        scatter_chart = charts_dir / "winner_board_alpha_vs_confidence.png"
        ordered_board = winner_board.sort_values(["winner_score", "ticker"], ascending=[False, True]).reset_index(drop=True)
        _write_scatter_chart(
            scatter_chart,
            x_values=ordered_board["pred_alpha_1d"].astype(float).tolist(),
            y_values=ordered_board["confidence_proxy"].astype(float).tolist(),
            labels=ordered_board["ticker"].astype(str).tolist(),
            colors=["#ff8c00" if bool(selected) else "#4f81bd" for selected in ordered_board.get("selected", pd.Series(False, index=ordered_board.index))],
            title="Predicted Alpha Versus Confidence Proxy",
            xlabel="pred_alpha_1d",
            ylabel="confidence_proxy",
        )
        chart_rows.append(
            {
                "chart_key": "winner_board_alpha_vs_confidence",
                "title": "Predicted Alpha Versus Confidence Proxy",
                "description": "Orange names are selected winners. Blue names were forecasted but not selected.",
                "relative_path": str(scatter_chart.relative_to(output_dir)).replace("\\", "/"),
            }
        )

    live_positions = live_boards.get("live_winner_positions_current", pd.DataFrame())
    if not live_positions.empty:
        realized_chart = charts_dir / "winner_predicted_vs_realized_alpha.png"
        _write_scatter_chart(
            realized_chart,
            x_values=live_positions["pred_alpha_1d"].astype(float).tolist(),
            y_values=live_positions["realized_alpha_1d"].astype(float).tolist(),
            labels=live_positions["ticker"].astype(str).tolist(),
            colors=["#2e8b57"] * len(live_positions),
            title="Winner Predicted Versus Realized Alpha",
            xlabel="pred_alpha_1d",
            ylabel="realized_alpha_1d",
        )
        chart_rows.append(
            {
                "chart_key": "winner_predicted_vs_realized_alpha",
                "title": "Winner Predicted Versus Realized Alpha",
                "description": "Available once the predicted session has been reconciled.",
                "relative_path": str(realized_chart.relative_to(output_dir)).replace("\\", "/"),
            }
        )

    charts = pd.DataFrame.from_records(chart_rows)
    if not charts.empty:
        charts.to_csv(output_dir / "chart_manifest.csv", index=False)
    return charts


def _render_winner_report_html(
    output_dir: Path,
    *,
    paper_summary: pd.DataFrame,
    prediction_summary: pd.DataFrame,
    winner_board: pd.DataFrame,
    winner_top_picks: pd.DataFrame,
    winner_daily_summary: pd.DataFrame,
    live_boards: dict[str, pd.DataFrame],
    backtest_boards: dict[str, pd.DataFrame],
    charts: pd.DataFrame,
) -> None:
    summary_cards: list[str] = []
    if not paper_summary.empty:
        row = paper_summary.iloc[0]
        for label, value in [
            ("Mode", row.get("report_mode", "")),
            ("Prediction Date", row.get("prediction_session", "")),
            ("Forecasted Names", row.get("n_forecasts", "")),
            ("Selected Winners", row.get("n_selected", "")),
            ("Model", row.get("model_id", "")),
            ("Context", row.get("context_length", "")),
            ("Basket Mode", row.get("basket_mode", "")),
            ("Beta Mode", row.get("beta_mode", "")),
            ("Confirmation", row.get("confirmation_mode", "")),
            ("Reconciliation", row.get("reconciliation_status", "")),
        ]:
            summary_cards.append(
                "<div class='card'><div class='label'>{}</div><div class='value'>{}</div></div>".format(
                    html.escape(str(label)),
                    html.escape(str(value)),
                )
            )

    chart_html = ""
    if not charts.empty:
        parts: list[str] = []
        for row in charts.itertuples(index=False):
            parts.append(
                "<div class='chart-block'><h3>{}</h3><p class='muted'>{}</p><img src='{}' alt='{}'></div>".format(
                    html.escape(str(row.title)),
                    html.escape(str(row.description)),
                    html.escape(str(row.relative_path)),
                    html.escape(str(row.title)),
                )
            )
        chart_html = "\n".join(parts)
    else:
        chart_html = "<p class='muted'>No charts were generated for this package.</p>"

    live_positions = live_boards.get("live_winner_positions_current", pd.DataFrame())
    live_daily = live_boards.get("live_winner_daily_summary_current", pd.DataFrame())
    live_summary = live_boards.get("live_winner_backtest_summary", pd.DataFrame())
    backtest_summary = backtest_boards.get("winner_backtest_summary", pd.DataFrame())

    html_body = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>Chronos Winner Report</title>
  <style>
    body {{
      font-family: Segoe UI, Arial, sans-serif;
      margin: 24px;
      color: #1f2933;
      background: #f8fafc;
    }}
    h1, h2, h3 {{
      color: #102a43;
    }}
    .muted {{
      color: #52606d;
    }}
    .cards {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
      gap: 12px;
      margin: 16px 0 24px;
    }}
    .card {{
      background: white;
      border: 1px solid #d9e2ec;
      border-radius: 10px;
      padding: 12px 14px;
      box-shadow: 0 1px 2px rgba(16, 42, 67, 0.08);
    }}
    .card .label {{
      font-size: 12px;
      text-transform: uppercase;
      color: #7b8794;
      margin-bottom: 6px;
    }}
    .card .value {{
      font-size: 18px;
      font-weight: 600;
      word-break: break-word;
    }}
    .chronos-table {{
      width: 100%;
      border-collapse: collapse;
      margin: 12px 0 20px;
      background: white;
      font-size: 13px;
    }}
    .chronos-table th, .chronos-table td {{
      border: 1px solid #d9e2ec;
      padding: 8px 10px;
      text-align: left;
      vertical-align: top;
    }}
    .chronos-table th {{
      background: #f0f4f8;
    }}
    .chart-block {{
      background: white;
      border: 1px solid #d9e2ec;
      border-radius: 10px;
      padding: 12px 14px;
      margin: 12px 0 18px;
    }}
    .chart-block img {{
      max-width: 100%;
      border-radius: 6px;
      border: 1px solid #d9e2ec;
      background: white;
    }}
  </style>
</head>
<body>
  <h1>Chronos Winner Picker Report</h1>
  <p class="muted">This package focuses on the additive winner-picker mode built on the locked Chronos-2 V2 core.</p>

  <h2>Overview</h2>
  <div class="cards">
    {''.join(summary_cards)}
  </div>

  <h2>Prediction Summary</h2>
  {_html_table(prediction_summary)}

  <h2>Winner Top Picks</h2>
  {_html_table(winner_top_picks)}

  <h2>Winner Board</h2>
  {_html_table(winner_board)}

  <h2>Daily Winner Summary</h2>
  {_html_table(winner_daily_summary)}

  <h2>Charts</h2>
  {chart_html}

  <h2>Live Reconciliation</h2>
  {_html_table(live_daily)}
  {_html_table(live_positions)}
  {_html_table(live_summary)}

  <h2>Historical Winner Snapshot</h2>
  {_html_table(backtest_summary)}
</body>
</html>
"""
    (output_dir / "paper_report.html").write_text(html_body, encoding="utf-8")


def _write_winner_notes(
    output_dir: Path,
    *,
    ledger_dir: Path | None = None,
    analysis_dir: Path | None = None,
) -> None:
    lines = [
        "Chronos winner-picker package guide",
        "",
        "winner_board.csv",
        "One row per forecasted ticker with winner-mode metrics such as pred_ret_1d, pred_alpha_1d, confidence_proxy, winner_score, and signal_label.",
        "",
        "winner_top_picks.csv",
        "Only the selected BUY and BUY_STRONG names ranked by winner_score.",
        "",
        "winner_daily_summary.csv",
        "One row per prediction day showing how many names were eligible, confirmed, and actually selected. Zero-pick days remain valid rows.",
        "",
        "confidence_proxy",
        "This is a heuristic strength ratio, not a calibrated probability.",
        "",
    ]
    if ledger_dir is not None:
        lines.extend(
            [
                "live_winner_positions.csv",
                "Ticker-level realized winner positions after reconciliation.",
                "",
                "live_winner_daily_summary.csv",
                "Daily realized winner-book summary after reconciliation.",
                "",
            ]
        )
    if analysis_dir is not None:
        lines.extend(
            [
                "winner_backtest_summary.csv",
                "Historical long-only winner summary from the frozen winner-mode run.",
                "",
            ]
        )
    (output_dir / "README.txt").write_text("\n".join(lines), encoding="utf-8")


def _render_paper_report_html(
    output_dir: Path,
    *,
    paper_summary: pd.DataFrame,
    prediction_summary_board: pd.DataFrame,
    prediction_board: pd.DataFrame,
    live_boards: dict[str, pd.DataFrame],
    backtest_boards: dict[str, pd.DataFrame],
    charts: pd.DataFrame,
) -> None:
    current_book = prediction_board[prediction_board["in_book"]].copy().sort_values(["side", "rank", "ticker"])
    top_universe = prediction_board.sort_values(["prediction_rank_desc", "ticker"]).copy()
    live_current_positions = live_boards.get("live_current_day_positions", pd.DataFrame())
    live_current_daily = live_boards.get("live_current_day_daily", pd.DataFrame())
    live_metrics_summary = live_boards.get("live_metrics_summary", pd.DataFrame())
    backtest_overview = backtest_boards.get("backtest_overview", pd.DataFrame())
    backtest_ticker_board = backtest_boards.get("backtest_ticker_board", pd.DataFrame())

    summary_cards: list[str] = []
    if not paper_summary.empty:
        row = paper_summary.iloc[0]
        for label, value in [
            ("Mode", row.get("report_mode", "")),
            ("Prediction Date", row.get("prediction_date", "")),
            ("Forecasted Names", row.get("n_forecasts", "")),
            ("Book Positions", row.get("n_book_positions", "")),
            ("Model", row.get("model_id", "")),
            ("Context", row.get("context_length", "")),
            ("Basket Mode", row.get("basket_mode", "")),
            ("Beta Mode", row.get("beta_mode", "")),
            ("Book Mode", row.get("book_mode", "")),
            ("Tail Fraction", row.get("tail_fraction", "")),
            ("Reconciliation", row.get("reconciliation_status", "")),
        ]:
            summary_cards.append(
                "<div class='card'><div class='label'>{}</div><div class='value'>{}</div></div>".format(
                    html.escape(str(label)),
                    html.escape(str(value)),
                )
            )

    chart_html = ""
    if not charts.empty:
        parts: list[str] = []
        for row in charts.itertuples(index=False):
            parts.append(
                "<div class='chart-block'><h3>{}</h3><p class='muted'>{}</p><img src='{}' alt='{}'></div>".format(
                    html.escape(str(row.title)),
                    html.escape(str(row.description)),
                    html.escape(str(row.relative_path)),
                    html.escape(str(row.title)),
                )
            )
        chart_html = "\n".join(parts)
    else:
        chart_html = "<p class='muted'>No charts were generated for this package.</p>"

    current_book_columns = [
        "ticker",
        "basket_id",
        "selection_status",
        "side",
        "rank",
        "score_plain",
        "score_z",
        "score_risk",
        "confidence_proxy",
        "beta_used",
        "alpha_uncertainty",
        "alpha_q50_path_text",
    ]
    universe_columns = [
        "ticker",
        "basket_id",
        "selection_status",
        "prediction_rank_desc",
        "confidence_rank_desc",
        "score_plain",
        "score_z",
        "confidence_proxy",
        "predicted_direction",
    ]
    live_position_columns = [
        "ticker",
        "basket_id",
        "side",
        "rank",
        "score_plain",
        "score_z",
        "next_day_realized_alpha",
        "gross_position_return",
        "book_impact",
    ]
    backtest_ticker_columns = [
        "ticker",
        "basket_id",
        "days_in_book",
        "hit_rate",
        "mean_abs_score",
        "mean_signed_realized_alpha",
        "total_gross_pnl",
        "contributor_rank",
        "drag_rank",
    ]

    html_body = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>Chronos Paper Report</title>
  <style>
    body {{
      font-family: Segoe UI, Arial, sans-serif;
      margin: 24px;
      color: #1f2933;
      background: #f8fafc;
    }}
    h1, h2, h3 {{
      color: #102a43;
    }}
    .muted {{
      color: #52606d;
    }}
    .cards {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
      gap: 12px;
      margin: 16px 0 24px;
    }}
    .card {{
      background: white;
      border: 1px solid #d9e2ec;
      border-radius: 10px;
      padding: 12px 14px;
      box-shadow: 0 1px 2px rgba(16, 42, 67, 0.08);
    }}
    .card .label {{
      font-size: 12px;
      text-transform: uppercase;
      color: #7b8794;
      margin-bottom: 6px;
    }}
    .card .value {{
      font-size: 18px;
      font-weight: 600;
      word-break: break-word;
    }}
    .chronos-table {{
      width: 100%;
      border-collapse: collapse;
      margin: 12px 0 20px;
      background: white;
      font-size: 13px;
    }}
    .chronos-table th, .chronos-table td {{
      border: 1px solid #d9e2ec;
      padding: 8px 10px;
      text-align: left;
      vertical-align: top;
    }}
    .chronos-table th {{
      background: #f0f4f8;
    }}
    .chart-block {{
      background: white;
      border: 1px solid #d9e2ec;
      border-radius: 10px;
      padding: 12px 14px;
      margin: 12px 0 18px;
    }}
    .chart-block img {{
      max-width: 100%;
      border-radius: 6px;
      border: 1px solid #d9e2ec;
      background: white;
    }}
    .file-list li {{
      margin: 6px 0;
    }}
  </style>
</head>
<body>
  <h1>Chronos Paper Report</h1>
  <p class="muted">This package combines the current prediction, the live paper-trading view when available, and the frozen backtest reference boards.</p>

  <h2>Overview</h2>
  <div class="cards">
    {''.join(summary_cards)}
  </div>

  <h2>Prediction Summary</h2>
  {_html_table(prediction_summary_board)}

  <h2>Current Tail Book</h2>
  <p class="muted">These are the names the model actually selected. Everything else was forecasted but not traded in the tail-only book.</p>
  {_html_table(current_book, columns=current_book_columns)}

  <h2>Prediction Universe</h2>
  <p class="muted">This table shows the ranked forecast universe so you can see what happened to the non-selected names too.</p>
  {_html_table(top_universe, columns=universe_columns)}

  <h2>Charts</h2>
  {chart_html}

  <h2>Live Reconciliation</h2>
  <p class="muted">If the predicted session has already finished and reconciliation succeeded, realized alpha and position impact appear here. Otherwise this section stays pending.</p>
  {_html_table(live_current_daily)}
  {_html_table(live_current_positions, columns=live_position_columns)}
  {_html_table(live_metrics_summary)}

  <h2>Historical Backtest Snapshot</h2>
  {_html_table(backtest_overview)}
  {_html_table(backtest_ticker_board, columns=backtest_ticker_columns, max_rows=12)}

  <h2>Files</h2>
  <ul class="file-list">
    <li><a href="prediction_board.csv">prediction_board.csv</a></li>
    <li><a href="prediction_slot_board.csv">prediction_slot_board.csv</a></li>
    <li><a href="prediction_summary_board.csv">prediction_summary_board.csv</a></li>
    <li><a href="paper_summary.csv">paper_summary.csv</a></li>
    <li><a href="README.txt">README.txt</a></li>
  </ul>
</body>
</html>
"""
    (output_dir / "paper_report.html").write_text(html_body, encoding="utf-8")


def write_chronos_notes(
    output_dir: Path,
    *,
    prediction_summary_board: pd.DataFrame,
    analysis_dir: Path | None,
    ledger_dir: Path | None = None,
) -> None:
    lines = [
        "Chronos package guide",
        "",
        "paper_report.html",
        "Single-file visual report that summarizes the current prediction, optional live reconciliation, and optional historical backtest reference.",
        "",
        "prediction_board.csv",
        "One row per forecasted ticker with explicit signal_session and prediction_session, plus score_plain, score_z, confidence_proxy, and in_book flag.",
        "",
        "prediction_slot_board.csv",
        "One row per ticker per forecast slot. Use this to see the path the model predicted across the next session.",
        "",
        "prediction_summary_board.csv",
        "One row per prediction_session with top longs, top shorts, confidence leader, and warnings about mixed sessions.",
        "",
        "Confidence definition",
        "confidence_proxy = abs(score_plain) / alpha_uncertainty. Higher means the predicted edge is large relative to the forecast interval width.",
        "",
        "Rank fields",
        "prediction_rank_desc ranks score_z from strongest to weakest within each prediction_session.",
        "confidence_rank_desc ranks confidence_proxy within each prediction_session.",
        "",
    ]
    if ledger_dir is not None:
        lines.extend(
            [
                "live_current_day_positions.csv",
                "Ticker-level realized live paper-trading results for the current prediction_session when reconciliation has completed.",
                "",
                "live_current_day_daily.csv",
                "One-row daily live scoreboard summary for the current prediction_session when reconciliation has completed.",
                "",
                "live_metrics_summary.csv",
                "Cumulative live paper-trading summary across all reconciled days in the selected ledger.",
                "",
            ]
        )
    if analysis_dir is not None:
        lines.extend(
            [
                "backtest_overview.csv",
                "High-level locked-baseline scorecard from the historical backtest.",
                "",
                "backtest_daily_board.csv",
                "One row per historical prediction day with daily spread, PnL, contributors, and Rank IC.",
                "",
                "backtest_position_board.csv",
                "One row per historical position with predicted score, realized alpha, contribution rank, and hit flag.",
                "",
                "backtest_ticker_board.csv",
                "Per-ticker contribution leaderboard across the historical backtest window.",
                "",
            ]
        )
    if not prediction_summary_board.empty and prediction_summary_board["signal_session_count"].max() > 1:
        lines.extend(
            [
                "Warning",
                "This package includes multiple signal sessions across rows. Read signal_session and prediction_session explicitly before comparing names.",
                "",
            ]
        )
    (output_dir / "README.txt").write_text("\n".join(lines), encoding="utf-8")


def export_chronos_package(
    *,
    prediction_dir: Path,
    output_dir: Path,
    analysis_dir: Path | None = None,
    ledger_dir: Path | None = None,
    report_mode: str = "report_only",
    report_context: dict[str, object] | None = None,
) -> None:
    import json

    with (prediction_dir / "forecasts.json").open("r", encoding="utf-8") as handle:
        forecasts = pd.DataFrame(json.load(handle))
    book = pd.read_csv(prediction_dir / "book.csv")
    prediction_summary = pd.read_csv(prediction_dir / "prediction_summary.csv")
    product_mode = _detect_product_mode(prediction_summary=prediction_summary, forecasts=forecasts)

    output_dir.mkdir(parents=True, exist_ok=True)
    if (prediction_dir / "data_quality_events.csv").exists():
        pd.read_csv(prediction_dir / "data_quality_events.csv").to_csv(output_dir / "prediction_data_quality_events.csv", index=False)

    normalized_prediction_summary = normalize_time_columns(
        prediction_summary,
        ["signal_session", "prediction_session", "prediction_session_end", "generated_at_utc"],
    )
    if not normalized_prediction_summary.empty and "prediction_session" in normalized_prediction_summary.columns:
        prediction_session = pd.to_datetime(
            normalized_prediction_summary["prediction_session"],
            utc=True,
            errors="coerce",
        ).max()
    else:
        prediction_session = None

    if product_mode == "winner_picker":
        winner_board_path = prediction_dir / "winner_board.csv"
        winner_top_picks_path = prediction_dir / "winner_top_picks.csv"
        winner_daily_summary_path = prediction_dir / "winner_daily_summary.csv"
        winner_daily_positions_path = prediction_dir / "winner_daily_positions.csv"

        winner_board = pd.read_csv(winner_board_path) if winner_board_path.exists() else pd.DataFrame()
        winner_top_picks = pd.read_csv(winner_top_picks_path) if winner_top_picks_path.exists() else pd.DataFrame()
        winner_daily_summary = pd.read_csv(winner_daily_summary_path) if winner_daily_summary_path.exists() else pd.DataFrame()
        winner_daily_positions = pd.read_csv(winner_daily_positions_path) if winner_daily_positions_path.exists() else pd.DataFrame()

        prediction_summary.to_csv(output_dir / "prediction_summary.csv", index=False)
        winner_board.to_csv(output_dir / "winner_board.csv", index=False)
        winner_top_picks.to_csv(output_dir / "winner_top_picks.csv", index=False)
        winner_daily_summary.to_csv(output_dir / "winner_daily_summary.csv", index=False)
        winner_daily_positions.to_csv(output_dir / "winner_daily_positions.csv", index=False)

        live_boards = build_live_report_boards(ledger_dir, prediction_session=prediction_session) if ledger_dir is not None else {}
        for name in [
            "live_winner_positions",
            "live_winner_positions_current",
            "live_winner_daily_summary",
            "live_winner_daily_summary_current",
            "live_winner_backtest_summary",
            "live_winner_rank_ic",
            "reconciliation_summary",
            "prediction_summary_history",
        ]:
            frame = live_boards.get(name)
            if frame is not None and not frame.empty:
                frame.to_csv(output_dir / f"{name}.csv", index=False)

        backtest_boards: dict[str, pd.DataFrame] = {}
        if analysis_dir is not None:
            for filename in ["winner_backtest_summary.csv"]:
                source = analysis_dir / filename
                if source.exists():
                    frame = pd.read_csv(source)
                    backtest_boards[filename.replace(".csv", "")] = frame
                    frame.to_csv(output_dir / filename, index=False)

        paper_summary = _build_winner_paper_summary(
            prediction_summary=prediction_summary,
            winner_board=winner_board,
            winner_top_picks=winner_top_picks,
            winner_daily_summary=winner_daily_summary,
            live_boards=live_boards,
            backtest_boards=backtest_boards,
            report_mode=report_mode,
            report_context=report_context,
        )
        paper_summary.to_csv(output_dir / "paper_summary.csv", index=False)
        charts = _write_winner_report_charts(
            output_dir,
            winner_board=winner_board,
            winner_top_picks=winner_top_picks,
            live_boards=live_boards,
        )
        _render_winner_report_html(
            output_dir,
            paper_summary=paper_summary,
            prediction_summary=prediction_summary,
            winner_board=winner_board,
            winner_top_picks=winner_top_picks,
            winner_daily_summary=winner_daily_summary,
            live_boards=live_boards,
            backtest_boards=backtest_boards,
            charts=charts,
        )
        _write_winner_notes(
            output_dir,
            ledger_dir=ledger_dir,
            analysis_dir=analysis_dir,
        )
        return

    prediction_board, prediction_slot_board, prediction_summary_board = build_prediction_board(
        forecasts,
        book,
        prediction_summary,
    )
    if prediction_session is None:
        prediction_session = (
            pd.to_datetime(prediction_summary_board["prediction_session"], utc=True, errors="coerce").max()
            if not prediction_summary_board.empty
            else None
        )
    prediction_board.to_csv(output_dir / "prediction_board.csv", index=False)
    prediction_slot_board.to_csv(output_dir / "prediction_slot_board.csv", index=False)
    prediction_summary_board.to_csv(output_dir / "prediction_summary_board.csv", index=False)

    live_boards = build_live_report_boards(ledger_dir, prediction_session=prediction_session) if ledger_dir is not None else {}
    for name in [
        "live_current_day_positions",
        "live_current_day_daily",
        "live_execution_current_day_positions",
        "live_execution_current_day_daily",
        "live_metrics_summary",
        "live_execution_sensitivity_comparison",
        "reconciliation_summary",
        "prediction_summary_history",
        "live_scoreboard_daily",
    ]:
        frame = live_boards.get(name)
        if frame is not None and not frame.empty:
            frame.to_csv(output_dir / f"{name}.csv", index=False)

    backtest_boards: dict[str, pd.DataFrame] = {}
    if analysis_dir is not None:
        backtest_boards = build_backtest_boards(analysis_dir)
        for name, frame in backtest_boards.items():
            frame.to_csv(output_dir / f"{name}.csv", index=False)
        for filename in [
            "signal_execution_baseline_summary.csv",
            "basket_book_comparison.csv",
            "side_book_comparison.csv",
            "execution_sensitivity_comparison.csv",
            "ticker_attribution_summary.csv",
            "oos_window_summary.csv",
            "oos_concentration_summary.csv",
            "side_isolation_summary.csv",
            "borrow_overlay_summary.csv",
            "borrow_overlay_ticker_summary.csv",
        ]:
            source = analysis_dir / filename
            if source.exists():
                pd.read_csv(source).to_csv(output_dir / filename, index=False)

    paper_summary = _build_paper_summary(
        prediction_summary_board=prediction_summary_board,
        prediction_board=prediction_board,
        live_boards=live_boards,
        backtest_boards=backtest_boards,
        report_mode=report_mode,
        report_context=report_context,
    )
    paper_summary.to_csv(output_dir / "paper_summary.csv", index=False)
    charts = _write_report_charts(
        output_dir,
        prediction_board=prediction_board,
        live_boards=live_boards,
        backtest_boards=backtest_boards,
    )
    _render_paper_report_html(
        output_dir,
        paper_summary=paper_summary,
        prediction_summary_board=prediction_summary_board,
        prediction_board=prediction_board,
        live_boards=live_boards,
        backtest_boards=backtest_boards,
        charts=charts,
    )
    write_chronos_notes(
        output_dir,
        prediction_summary_board=prediction_summary_board,
        analysis_dir=analysis_dir,
        ledger_dir=ledger_dir,
    )
