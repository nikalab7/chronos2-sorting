from __future__ import annotations

import argparse
from datetime import date, timedelta
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from chronos_equity_prototype.config import (  # noqa: E402
    DEFAULT_LIQUID_SP500,
    DEFAULT_PRODUCT_MODE,
    DEFAULT_SHORT_BORROW_FEE_ANNUAL_BPS,
    DEFAULT_V2_BETA_MODE,
    DEFAULT_V2_BOOK_MODE,
    DEFAULT_V2_TAIL_FRACTION,
    DEFAULT_WINNER_CONFIRMATION_MODES,
    DEFAULT_WINNER_CONFIRMATION_MODE,
    ForecasterConfig,
    ResearchConfig,
)
from chronos_equity_prototype.pipeline import ZeroShotResearchPipeline  # noqa: E402
from chronos_equity_prototype.reporting import export_chronos_package  # noqa: E402
from chronos_equity_prototype.v2_basket import BasketV2Builder  # noqa: E402


LOCKED_BASELINE_TEXT = [
    "Locked baseline",
    "  architecture   : v2_flagship_basket_raw",
    "  model          : amazon/chronos-2",
    "  context_length : 224",
    "  basket_mode    : expanded40",
    "  beta_mode      : unit_beta_1.0",
    "  ranking_score  : score_z",
    "  book_mode      : tail_fraction",
    "  tail_fraction  : 0.10",
]


def _model_slug(model_id: str) -> str:
    slug = model_id.strip().lower().replace("/", "_").replace("-", "_")
    return "".join(char if char.isalnum() or char == "_" else "_" for char in slug)


def _default_backtest_window() -> tuple[str, str]:
    end = date.today()
    start = end - timedelta(days=59)
    return start.isoformat(), end.isoformat()


def _default_prediction_window() -> tuple[str, str]:
    end = date.today() + timedelta(days=1)
    start = end - timedelta(days=59)
    return start.isoformat(), end.isoformat()


def _default_output_dir(mode: str) -> Path:
    return ROOT / "outputs" / f"workstation_{mode}_{date.today().isoformat()}"


def _outputs_root() -> Path:
    return ROOT / "outputs"


def _parse_tickers(tickers: str | None, tickers_file: str | None) -> list[str]:
    if tickers:
        return [ticker.strip().upper() for ticker in tickers.split(",") if ticker.strip()]
    if tickers_file:
        content = Path(tickers_file).read_text(encoding="utf-8")
        return [line.strip().upper() for line in content.splitlines() if line.strip()]
    return list(DEFAULT_LIQUID_SP500)


def _resolve_v2_tickers(base_tickers: list[str], architecture: str, basket_mode: str) -> list[str]:
    if architecture != "v2_flagship_basket_raw":
        return base_tickers
    basket_tickers = BasketV2Builder().required_tickers(basket_mode)
    return list(dict.fromkeys([*base_tickers, *basket_tickers]))


def _resolve_context_length(args: argparse.Namespace) -> int:
    if args.context_length is not None:
        return args.context_length
    if args.architecture == "v2_flagship_basket_raw":
        return ResearchConfig().default_v2_context_length
    return ResearchConfig().default_context_length


def _build_experiment_name(args: argparse.Namespace, context_length: int) -> str:
    model_slug = _model_slug(args.model_id)
    if args.architecture == "v2_flagship_basket_raw":
        if getattr(args, "product_mode", DEFAULT_PRODUCT_MODE) == "winner_picker":
            return (
                f"{args.architecture}_{args.basket_mode}_{args.product_mode}_{model_slug}_ctx{context_length}_"
                f"confirm{args.confirmation_mode}_cross{int(args.cross_learning)}_cov{int(not args.no_covariates)}_"
                f"beta{args.beta_mode}"
            )
        return (
            f"{args.architecture}_{args.basket_mode}_{model_slug}_ctx{context_length}_"
            f"cross{int(args.cross_learning)}_cov{int(not args.no_covariates)}"
        )
    if getattr(args, "product_mode", DEFAULT_PRODUCT_MODE) != "ranker":
        return (
            f"{args.architecture}_{args.product_mode}_{model_slug}_ctx{context_length}_"
            f"cross{int(args.cross_learning)}_cov{int(not args.no_covariates)}"
        )
    return (
        f"{args.architecture}_{model_slug}_ctx{context_length}_"
        f"cross{int(args.cross_learning)}_cov{int(not args.no_covariates)}"
    )


def _build_pipeline(args: argparse.Namespace, tickers: list[str]) -> tuple[ZeroShotResearchPipeline, int]:
    resolved_context_length = _resolve_context_length(args)
    research_defaults = ResearchConfig()
    research_config = ResearchConfig(
        model_id=args.model_id,
        prediction_length=(
            research_defaults.winner_prediction_length
            if getattr(args, "product_mode", DEFAULT_PRODUCT_MODE) == "winner_picker"
            else args.prediction_length
        ),
        default_context_length=resolved_context_length,
        benchmark_ticker=args.benchmark.upper(),
        batch_size=args.batch_size,
        n_long=args.n_long,
        n_short=args.n_short,
        liquid_universe=tuple(tickers),
    )
    return ZeroShotResearchPipeline(research_config=research_config), resolved_context_length


def _build_forecaster_config(args: argparse.Namespace, context_length: int) -> ForecasterConfig:
    defaults = ResearchConfig()
    product_mode = getattr(args, "product_mode", DEFAULT_PRODUCT_MODE)
    return ForecasterConfig(
        model_id=args.model_id,
        prediction_length=defaults.winner_prediction_length if product_mode == "winner_picker" else args.prediction_length,
        context_length=context_length,
        quantiles=defaults.winner_quantiles if product_mode == "winner_picker" else defaults.quantiles,
        cross_learning=args.cross_learning,
        use_covariates=not args.no_covariates,
        batch_size=args.batch_size,
        beta_mode=args.beta_mode,
        product_mode=product_mode,
        confirmation_mode=getattr(args, "confirmation_mode", DEFAULT_WINNER_CONFIRMATION_MODE),
    )


def _print_live_summary(prediction_dir: Path, report_dir: Path, reconcile_status: str, reconcile_message: str) -> None:
    summary = pd.read_csv(prediction_dir / "prediction_summary.csv")
    book = pd.read_csv(prediction_dir / "book.csv")
    summary_row = summary.iloc[0]
    product_mode = str(summary_row.get("product_mode", "ranker"))

    print()
    print("Live prediction summary")
    print(f"  Signal session:     {pd.Timestamp(summary_row['signal_session']).date()}")
    print(f"  Prediction session: {pd.Timestamp(summary_row['prediction_session']).date()}")
    print(f"  Forecast universe:  {int(summary_row['n_forecasts'])}")
    print(f"  Longs / shorts:     {int(summary_row['n_longs'])} / {int(summary_row['n_shorts'])}")
    print(f"  Model:              {summary_row['model_id']}")
    print(f"  Context length:     {int(summary_row['context_length'])}")
    if bool(summary_row.get("has_newer_observed_session_than_signal", False)):
        observed_latest = summary_row.get("observed_latest_session", "")
        observed_text = str(pd.Timestamp(observed_latest).date()) if pd.notna(observed_latest) and observed_latest else "unknown"
        print(f"  Freshness:          WARNING newer observed session {observed_text} was present")
    elif summary_row.get("prediction_freshness", ""):
        print(f"  Freshness:          {summary_row['prediction_freshness']}")
    print(f"  Reconciliation:     {reconcile_status}")
    if reconcile_message:
        print(f"  Reconcile note:     {reconcile_message}")
    print()
    if product_mode == "winner_picker":
        winner_top_picks_path = prediction_dir / "winner_top_picks.csv"
        winner_board_path = prediction_dir / "winner_board.csv"
        preview = pd.read_csv(winner_top_picks_path) if winner_top_picks_path.exists() else pd.DataFrame()
        if preview.empty and winner_board_path.exists():
            preview = pd.read_csv(winner_board_path).head(12)
        preview_columns = [
            column
            for column in [
                "rank",
                "ticker",
                "basket_id",
                "winner_score",
                "pred_ret_1d",
                "pred_alpha_1d",
                "pred_ret_3d",
                "pred_alpha_3d",
                "confidence_proxy",
                "signal_label",
            ]
            if column in preview.columns
        ]
        if preview.empty:
            winner_daily_summary_path = prediction_dir / "winner_daily_summary.csv"
            if winner_daily_summary_path.exists():
                print(pd.read_csv(winner_daily_summary_path).to_string(index=False))
            else:
                print("No winner-mode candidates were written.")
        else:
            print(preview[preview_columns].to_string(index=False))
    else:
        preview_columns = [
            column
            for column in ["side", "rank", "ticker", "basket_id", "score_plain", "score_z"]
            if column in book.columns
        ]
        preview = book[preview_columns].sort_values(["side", "rank", "ticker"])
        print(preview.to_string(index=False))
    print()
    print(f"Prediction files: {prediction_dir}")
    print(f"Paper report:     {report_dir / 'paper_report.html'}")


def _read_csv_if_exists(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def _print_section(title: str) -> None:
    print()
    print(title)
    print("-" * len(title))


def _print_dataframe(frame: pd.DataFrame, *, columns: list[str] | None = None, max_rows: int | None = None) -> None:
    if frame.empty:
        print("No rows available.")
        return
    table = frame.copy()
    if columns is not None:
        available = [column for column in columns if column in table.columns]
        table = table[available]
    if max_rows is not None:
        table = table.head(max_rows)
    print(table.to_string(index=False))


def _print_terminal_report(report_dir: Path) -> None:
    paper_summary = _read_csv_if_exists(report_dir / "paper_summary.csv")
    prediction_board = _read_csv_if_exists(report_dir / "prediction_board.csv")
    winner_board = _read_csv_if_exists(report_dir / "winner_board.csv")
    winner_top_picks = _read_csv_if_exists(report_dir / "winner_top_picks.csv")
    winner_daily_summary = _read_csv_if_exists(report_dir / "winner_daily_summary.csv")
    live_winner_positions = _read_csv_if_exists(report_dir / "live_winner_positions.csv")
    live_winner_daily_summary = _read_csv_if_exists(report_dir / "live_winner_daily_summary.csv")
    live_winner_backtest_summary = _read_csv_if_exists(report_dir / "live_winner_backtest_summary.csv")
    live_current_day_positions = _read_csv_if_exists(report_dir / "live_current_day_positions.csv")
    live_current_day_daily = _read_csv_if_exists(report_dir / "live_current_day_daily.csv")
    live_metrics_summary = _read_csv_if_exists(report_dir / "live_metrics_summary.csv")
    backtest_overview = _read_csv_if_exists(report_dir / "backtest_overview.csv")
    winner_backtest_summary = _read_csv_if_exists(report_dir / "winner_backtest_summary.csv")
    has_detailed_backtest_tables = any(
        (report_dir / filename).exists()
        for filename in [
            "signal_execution_baseline_summary.csv",
            "basket_book_comparison.csv",
            "side_book_comparison.csv",
            "execution_sensitivity_comparison.csv",
            "ticker_attribution_summary.csv",
        ]
    )

    if not paper_summary.empty:
        row = paper_summary.iloc[0]
        _print_section("Run Overview")
        print(f"Prediction date:        {row.get('prediction_date', '')}")
        print(f"Forecast universe:      {row.get('n_forecasts', '')}")
        print(f"Selected book size:     {row.get('n_book_positions', '')}")
        print(f"Top score ticker:       {row.get('universe_top_score_ticker', '')}")
        print(f"Top confidence ticker:  {row.get('universe_top_confidence_ticker', '')}")
        if bool(row.get("has_newer_observed_session_than_signal", False)):
            observed_latest = row.get("observed_latest_session", "")
            observed_text = str(pd.Timestamp(observed_latest).date()) if pd.notna(observed_latest) and observed_latest else "unknown"
            print(f"Freshness warning:      newer observed session {observed_text} was present in fetched data")
        elif row.get("prediction_freshness", ""):
            print(f"Prediction freshness:   {row.get('prediction_freshness', '')}")
        print(f"Reconciliation status:  {row.get('reconciliation_status', '')}")

    if not winner_board.empty:
        _print_section("Winner Board")
        _print_dataframe(
            winner_board.sort_values(["winner_score", "ticker"], ascending=[False, True]),
            columns=[
                "ticker",
                "basket_id",
                "winner_score",
                "pred_ret_1d",
                "pred_alpha_1d",
                "pred_ret_3d",
                "pred_alpha_3d",
                "confidence_proxy",
                "signal_label",
                "buy_eligible",
                "confirmed_3d",
            ],
        )

    if not winner_top_picks.empty:
        _print_section("Winner Top Picks")
        _print_dataframe(
            winner_top_picks.sort_values(["rank", "ticker"]),
            columns=[
                "rank",
                "ticker",
                "basket_id",
                "winner_score",
                "pred_ret_1d",
                "pred_alpha_1d",
                "confidence_proxy",
                "signal_label",
            ],
        )

    if not winner_daily_summary.empty:
        _print_section("Winner Daily Summary")
        _print_dataframe(
            winner_daily_summary,
            columns=[
                "prediction_session",
                "n_candidates",
                "n_buy_eligible",
                "n_confirmed_3d",
                "n_selected",
                "book_active",
                "selected_tickers",
            ],
        )

    if not prediction_board.empty:
        _print_section("Ticker Forecasts")
        _print_dataframe(
            prediction_board.sort_values(["prediction_rank_desc", "ticker"]),
            columns=[
                "ticker",
                "basket_id",
                "selection_status",
                "side",
                "rank",
                "prediction_rank_desc",
                "confidence_rank_desc",
                "score_plain",
                "score_z",
                "score_risk",
                "confidence_proxy",
                "alpha_uncertainty",
                "predicted_direction",
            ],
        )

    if not live_winner_daily_summary.empty:
        _print_section("Live Winner Reconciliation")
        _print_dataframe(
            live_winner_daily_summary,
            columns=[
                "prediction_session",
                "n_selected",
                "book_active",
                "gross_return",
                "mean_selected_return",
                "mean_selected_alpha",
                "best_contributor",
                "worst_contributor",
            ],
        )

    if not live_winner_positions.empty:
        _print_section("Live Winner Positions")
        _print_dataframe(
            live_winner_positions,
            columns=[
                "ticker",
                "rank",
                "winner_score",
                "pred_ret_1d",
                "pred_alpha_1d",
                "realized_stock_ret_1d",
                "realized_alpha_1d",
                "gross_position_return",
                "book_impact",
            ],
        )

    if not live_winner_backtest_summary.empty:
        _print_section("Live Winner Metrics")
        _print_dataframe(live_winner_backtest_summary)

    if not live_current_day_positions.empty:
        _print_section("Current Realized Positions")
        _print_dataframe(
            live_current_day_positions.sort_values(["side", "rank", "ticker"]),
            columns=[
                "ticker",
                "basket_id",
                "side",
                "rank",
                "score_plain",
                "score_z",
                "next_day_realized_alpha",
                "gross_position_return",
                "book_impact",
            ],
        )

    if not live_current_day_daily.empty:
        _print_section("Current Book Result")
        _print_dataframe(
            live_current_day_daily,
            columns=[
                "prediction_date",
                "long_basket_return",
                "short_basket_return",
                "long_short_spread",
                "gross_long_short_pnl",
                "best_contributor",
                "worst_contributor",
                "growth_result_gross_pnl",
                "defensive_result_gross_pnl",
            ],
        )

    if not live_metrics_summary.empty:
        _print_section("Live Regime Metrics")
        _print_dataframe(
            live_metrics_summary,
            columns=[
                "cost_bps",
                "return_type",
                "mean_daily_return",
                "daily_volatility",
                "annualized_sharpe",
                "win_rate",
            ],
        )

    if not winner_backtest_summary.empty:
        _print_section("Winner Backtest Metrics")
        _print_dataframe(winner_backtest_summary)

    if has_detailed_backtest_tables:
        _print_backtest_analysis_tables(report_dir)
    elif not backtest_overview.empty:
        _print_section("Historical Regime Metrics")
        overview = backtest_overview.iloc[0]
        regime_rows = pd.DataFrame(
            [
                {
                    "regime": "overall",
                    "mean_rank_ic": overview.get("mean_rank_ic"),
                    "positive_rank_ic_rate": overview.get("positive_rank_ic_rate"),
                    "gross_sharpe": overview.get("gross_sharpe"),
                    "net_sharpe_10bps": overview.get("net_sharpe_10bps"),
                    "net_sharpe_20bps": overview.get("net_sharpe_20bps"),
                    "net_sharpe_40bps": overview.get("net_sharpe_40bps"),
                    "long_hit_rate": overview.get("long_hit_rate"),
                    "short_hit_rate": overview.get("short_hit_rate"),
                    "book_hit_rate": overview.get("book_hit_rate"),
                },
                {
                    "regime": "growth_only",
                    "mean_rank_ic": np.nan,
                    "positive_rank_ic_rate": np.nan,
                    "gross_sharpe": np.nan,
                    "net_sharpe_10bps": overview.get("growth_only_net_sharpe_10bps"),
                    "net_sharpe_20bps": np.nan,
                    "net_sharpe_40bps": np.nan,
                    "long_hit_rate": np.nan,
                    "short_hit_rate": np.nan,
                    "book_hit_rate": np.nan,
                },
                {
                    "regime": "defensive_only",
                    "mean_rank_ic": np.nan,
                    "positive_rank_ic_rate": np.nan,
                    "gross_sharpe": np.nan,
                    "net_sharpe_10bps": overview.get("defensive_only_net_sharpe_10bps"),
                    "net_sharpe_20bps": np.nan,
                    "net_sharpe_40bps": np.nan,
                    "long_hit_rate": np.nan,
                    "short_hit_rate": np.nan,
                    "book_hit_rate": np.nan,
                },
                {
                    "regime": "exact_open",
                    "mean_rank_ic": np.nan,
                    "positive_rank_ic_rate": np.nan,
                    "gross_sharpe": np.nan,
                    "net_sharpe_10bps": overview.get("exact_open_net_sharpe_10bps"),
                    "net_sharpe_20bps": overview.get("exact_open_net_sharpe_20bps"),
                    "net_sharpe_40bps": overview.get("exact_open_net_sharpe_40bps"),
                    "long_hit_rate": np.nan,
                    "short_hit_rate": np.nan,
                    "book_hit_rate": np.nan,
                },
                {
                    "regime": "first_5m_avg",
                    "mean_rank_ic": np.nan,
                    "positive_rank_ic_rate": np.nan,
                    "gross_sharpe": np.nan,
                    "net_sharpe_10bps": overview.get("first_5m_avg_net_sharpe_10bps"),
                    "net_sharpe_20bps": overview.get("first_5m_avg_net_sharpe_20bps"),
                    "net_sharpe_40bps": overview.get("first_5m_avg_net_sharpe_40bps"),
                    "long_hit_rate": np.nan,
                    "short_hit_rate": np.nan,
                    "book_hit_rate": np.nan,
                },
                {
                    "regime": "first_15m_avg",
                    "mean_rank_ic": np.nan,
                    "positive_rank_ic_rate": np.nan,
                    "gross_sharpe": np.nan,
                    "net_sharpe_10bps": overview.get("first_15m_avg_net_sharpe_10bps"),
                    "net_sharpe_20bps": overview.get("first_15m_avg_net_sharpe_20bps"),
                    "net_sharpe_40bps": overview.get("first_15m_avg_net_sharpe_40bps"),
                    "long_hit_rate": np.nan,
                    "short_hit_rate": np.nan,
                    "book_hit_rate": np.nan,
                },
            ]
        )
        _print_dataframe(regime_rows)
        print()
        print(f"Top historical contributors: {overview.get('top_contributors', '')}")
        print(f"Top historical drags:        {overview.get('top_drags', '')}")


def _print_backtest_analysis_tables(analysis_dir: Path) -> None:
    baseline_summary = _read_csv_if_exists(analysis_dir / "signal_execution_baseline_summary.csv")
    basket_book_comparison = _read_csv_if_exists(analysis_dir / "basket_book_comparison.csv")
    side_book_comparison = _read_csv_if_exists(analysis_dir / "side_book_comparison.csv")
    execution_sensitivity = _read_csv_if_exists(analysis_dir / "execution_sensitivity_comparison.csv")
    ticker_attribution = _read_csv_if_exists(analysis_dir / "ticker_attribution_summary.csv")
    oos_window_summary = _read_csv_if_exists(analysis_dir / "oos_window_summary.csv")
    oos_concentration_summary = _read_csv_if_exists(analysis_dir / "oos_concentration_summary.csv")
    side_isolation_summary = _read_csv_if_exists(analysis_dir / "side_isolation_summary.csv")
    borrow_overlay_summary = _read_csv_if_exists(analysis_dir / "borrow_overlay_summary.csv")
    borrow_overlay_ticker_summary = _read_csv_if_exists(analysis_dir / "borrow_overlay_ticker_summary.csv")

    if not baseline_summary.empty:
        _print_section("Backtest Baseline")
        _print_dataframe(
            baseline_summary,
            columns=[
                "score_column",
                "prediction_days",
                "mean_rank_ic",
                "positive_rank_ic_rate",
                "gross_sharpe",
                "net_sharpe_10bps",
                "net_sharpe_20bps",
                "net_sharpe_40bps",
                "long_hit_rate",
                "short_hit_rate",
                "top1_capture_mean",
                "top5_capture_mean",
            ],
        )

    if not oos_window_summary.empty:
        _print_section("OOS Window")
        _print_dataframe(
            oos_window_summary,
            columns=[
                "complete_sessions",
                "context_length",
                "required_context_sessions",
                "current_prediction_days",
                "theoretical_max_prediction_days",
                "target_gap_20d",
                "target_gap_30d",
            ],
        )

    if not oos_concentration_summary.empty:
        _print_section("OOS Concentration")
        _print_dataframe(
            oos_concentration_summary,
            columns=[
                "prediction_days",
                "positive_days",
                "negative_days",
                "best_day_abs_pnl_share",
                "top2_abs_pnl_share",
                "best_day_gross_pnl",
                "worst_day_gross_pnl",
            ],
        )

    if not basket_book_comparison.empty:
        _print_section("Basket Regimes")
        _print_dataframe(
            basket_book_comparison,
            columns=[
                "basket_slice",
                "prediction_days",
                "mean_daily_return",
                "gross_sharpe",
                "net_sharpe_10bps",
                "net_sharpe_20bps",
                "net_sharpe_40bps",
                "hit_rate",
                "contribution_share",
            ],
        )

    if not side_book_comparison.empty:
        _print_section("Side Regimes")
        _print_dataframe(
            side_book_comparison,
            columns=[
                "side_slice",
                "prediction_days",
                "mean_daily_return",
                "gross_sharpe",
                "net_sharpe_10bps",
                "net_sharpe_20bps",
                "net_sharpe_40bps",
                "hit_rate",
            ],
        )

    if not side_isolation_summary.empty:
        _print_section("Side Isolation")
        _print_dataframe(
            side_isolation_summary,
            columns=[
                "side",
                "prediction_days",
                "mean_daily_gross_pnl",
                "hit_rate",
                "total_gross_pnl",
                "top_ticker",
                "top_ticker_abs_pnl_share",
                "top2_ticker_abs_pnl_share",
            ],
        )

    if not execution_sensitivity.empty:
        _print_section("Execution Regimes")
        _print_dataframe(
            execution_sensitivity,
            columns=[
                "execution_mode",
                "prediction_days",
                "mean_daily_return",
                "gross_sharpe",
                "net_sharpe_10bps",
                "net_sharpe_20bps",
                "net_sharpe_40bps",
                "hit_rate",
            ],
        )

    if not borrow_overlay_summary.empty:
        _print_section("Borrow Overlay")
        _print_dataframe(
            borrow_overlay_summary,
            columns=[
                "prediction_days",
                "gross_sharpe",
                "net_sharpe_10bps",
                "net_sharpe_20bps",
                "net_sharpe_40bps",
                "mean_daily_borrow_cost",
                "excluded_short_positions",
                "excluded_prediction_days",
                "mean_short_borrow_fee_annual_bps",
            ],
        )

    if not ticker_attribution.empty:
        _print_section("Top Ticker Contributors")
        _print_dataframe(
            ticker_attribution.sort_values(["total_gross_pnl", "ticker"], ascending=[False, True]),
            columns=[
                "ticker",
                "basket_id",
                "days_in_book",
                "hit_rate",
                "mean_abs_score",
                "mean_signed_realized_alpha",
                "total_gross_pnl",
            ],
            max_rows=15,
        )

    if not borrow_overlay_ticker_summary.empty:
        _print_section("Borrow-Adjusted Shorts")
        _print_dataframe(
            borrow_overlay_ticker_summary.sort_values(["borrow_pnl_drag", "ticker"], ascending=[False, True]),
            columns=[
                "ticker",
                "basket_id",
                "short_days",
                "excluded_short_days",
                "mean_borrow_fee_annual_bps",
                "borrow_pnl_drag",
                "total_gross_pnl_after_borrow",
            ],
            max_rows=15,
        )


def _print_backtest_summary(experiment_dir: Path, analysis_dir: Path | None) -> None:
    winner_backtest_summary = _read_csv_if_exists(experiment_dir / "winner_backtest_summary.csv")
    if not winner_backtest_summary.empty:
        print()
        print("Winner backtest summary")
        _print_dataframe(winner_backtest_summary)
        print()
        print(f"Experiment files: {experiment_dir}")
        if analysis_dir is not None:
            print(f"Analysis files:   {analysis_dir}")
        return

    metrics = pd.read_csv(experiment_dir / "metrics_summary.csv")
    rank_ic = pd.read_csv(experiment_dir / "rank_ic.csv")
    mean_rank_ic = rank_ic.loc[rank_ic["metric"] == "mean_rank_ic", "value"]
    positive_rank_ic = rank_ic.loc[rank_ic["metric"] == "positive_rank_ic_rate", "value"]
    print()
    print("Backtest summary")
    if not mean_rank_ic.empty:
        print(f"  Mean Rank IC:       {float(mean_rank_ic.iloc[0]):.4f}")
    if not positive_rank_ic.empty:
        print(f"  Positive Rank IC:   {float(positive_rank_ic.iloc[0]):.2%}")
    for cost_bps in [0, 10, 20, 40]:
        row = metrics.loc[metrics["cost_bps"] == cost_bps]
        if row.empty:
            continue
        return_type = str(row.iloc[0]["return_type"])
        sharpe = row.iloc[0].get("annualized_sharpe")
        mean_daily = row.iloc[0].get("mean_daily_return")
        print(f"  {cost_bps:>2} bps {return_type:<4} Sharpe: {sharpe} | mean daily return: {mean_daily}")
    print()
    print(f"Experiment files: {experiment_dir}")
    if analysis_dir is not None:
        print(f"Analysis files:   {analysis_dir}")
        _print_backtest_analysis_tables(analysis_dir)


def _latest_matching_parent(filename: str, *, must_contain: tuple[str, ...] = ()) -> Path | None:
    outputs_root = _outputs_root()
    if not outputs_root.exists():
        return None
    candidates: list[Path] = []
    for path in outputs_root.rglob(filename):
        parent = path.parent
        if all((parent / item).exists() for item in must_contain):
            candidates.append(parent)
    if not candidates:
        return None
    return max(candidates, key=lambda parent: parent.stat().st_mtime)


def _default_analysis_dir() -> str:
    explicit = _outputs_root() / "fresh_locked_v2_expanded40_analysis"
    if explicit.exists():
        return str(explicit)
    latest = _latest_matching_parent("backtest_scorecard.csv")
    return str(latest) if latest is not None else ""


def _default_prediction_dir() -> str:
    latest = _latest_matching_parent(
        "prediction_summary.csv",
        must_contain=("book.csv", "forecasts.json"),
    )
    return str(latest) if latest is not None else ""


def _default_ledger_dir(basket_mode: str = "expanded40") -> str:
    explicit = _outputs_root() / f"live_scoreboard_{basket_mode}"
    if explicit.exists():
        return str(explicit)
    latest = _latest_matching_parent("reconciliation_summary.csv")
    return str(latest) if latest is not None else str(explicit)


def _normalize_analysis_tail_fractions(value: object) -> list[float]:
    if isinstance(value, (list, tuple)):
        return [float(item) for item in value]
    return [float(item.strip()) for item in str(value).split(",") if item.strip()]


def _normalize_optional_tickers(value: object) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, (list, tuple)):
        return tuple(str(item).strip().upper() for item in value if str(item).strip())
    return tuple(item.strip().upper() for item in str(value).split(",") if item.strip())


def _print_workstation_banner() -> None:
    print()
    print("Chronos Workstation")
    print("===================")
    for line in LOCKED_BASELINE_TEXT:
        print(line)
    print()
    print("Menu")
    print("  1. Live prediction with paper report")
    print("  2. Backtest with analysis")
    print("  3. Build report from existing prediction")
    print("  4. Exit")


def _prompt_text(label: str, default: str = "") -> str:
    suffix = f" [{default}]" if default else ""
    value = input(f"{label}{suffix}: ").strip()
    return value or default


def _prompt_yes_no(label: str, default: bool = True) -> bool:
    default_label = "Y/n" if default else "y/N"
    value = input(f"{label} [{default_label}]: ").strip().lower()
    if not value:
        return default
    return value in {"y", "yes"}


def _interactive_namespace(mode: str) -> argparse.Namespace:
    parser = build_parser()
    namespace = parser.parse_args(["--mode", mode])
    namespace.output_dir = None
    namespace.prediction_dir = None
    namespace.analysis_dir = None
    namespace.ledger_dir = None
    namespace.analysis_tail_fractions = _normalize_analysis_tail_fractions(namespace.analysis_tail_fractions)
    return namespace


def _run_interactive_menu() -> None:
    while True:
        _print_workstation_banner()
        choice = input("Select option [1]: ").strip() or "1"
        print()

        if choice == "1":
            args = _interactive_namespace("live")
            default_output = str(_default_output_dir("live"))
            default_analysis = _default_analysis_dir()
            default_ledger = _default_ledger_dir(args.basket_mode)
            args.product_mode = _prompt_text("Product mode", args.product_mode)
            if args.product_mode == "winner_picker":
                args.confirmation_mode = _prompt_text("Winner confirmation mode", args.confirmation_mode)
            args.output_dir = _prompt_text("Output folder", default_output)
            args.analysis_dir = _prompt_text("Frozen backtest analysis folder", default_analysis)
            args.ledger_dir = _prompt_text("Live scoreboard folder", default_ledger)
            run_live_mode(args)
        elif choice == "2":
            args = _interactive_namespace("backtest")
            backtest_start, backtest_end = _default_backtest_window()
            args.product_mode = _prompt_text("Product mode", args.product_mode)
            if args.product_mode == "winner_picker":
                args.confirmation_mode = _prompt_text("Winner confirmation mode", args.confirmation_mode)
            args.output_dir = _prompt_text("Output folder", str(_default_output_dir("backtest")))
            args.start = _prompt_text("Start date", backtest_start)
            args.end = _prompt_text("End date", backtest_end)
            args.skip_analysis = not _prompt_yes_no("Run analyze-signal after backtest", True)
            run_backtest_mode(args)
        elif choice == "3":
            args = _interactive_namespace("report")
            prediction_default = _default_prediction_dir()
            if not prediction_default:
                print("No prediction directory was found under outputs yet.")
                print("Run live mode first, or provide a prediction folder manually with command-line flags.")
            else:
                args.prediction_dir = _prompt_text("Prediction directory", prediction_default)
                args.analysis_dir = _prompt_text("Frozen backtest analysis folder", _default_analysis_dir())
                args.ledger_dir = _prompt_text("Live scoreboard folder", _default_ledger_dir())
                args.output_dir = _prompt_text("Output folder", str(_default_output_dir("report")))
                run_report_mode(args)
        elif choice == "4":
            print("Exiting Chronos Workstation.")
            return
        else:
            print("Unknown option. Choose 1, 2, 3, or 4.")

        print()
        input("Press Enter to return to the main menu...")


def run_live_mode(args: argparse.Namespace) -> None:
    start, end = _default_prediction_window()
    if args.start:
        start = args.start
    if args.end:
        end = args.end

    base_tickers = _parse_tickers(args.tickers, args.tickers_file)
    tickers = _resolve_v2_tickers(base_tickers, args.architecture, args.basket_mode)
    if args.product_mode == "winner_picker" and args.architecture != "v2_flagship_basket_raw":
        print("winner_picker is supported only with v2_flagship_basket_raw.")
        return
    pipeline, context_length = _build_pipeline(args, tickers)
    forecaster_config = _build_forecaster_config(args, context_length)
    experiment_name = _build_experiment_name(args, context_length)

    output_dir = Path(args.output_dir) if args.output_dir else _default_output_dir("live")
    prediction_root = output_dir / "prediction"
    report_dir = output_dir / "report"
    ledger_dir = Path(args.ledger_dir) if args.ledger_dir else ROOT / "outputs" / f"live_scoreboard_{args.basket_mode}"
    output_dir.mkdir(parents=True, exist_ok=True)
    prediction_root.mkdir(parents=True, exist_ok=True)

    if args.architecture == "v2_flagship_basket_raw":
        labeled_frame, v2_base_frame = pipeline.prepare_market_frames(tickers=tickers, start=start, end=end)
        labeled_frame.to_csv(output_dir / "labeled_frame.csv", index=False)
        v2_base_frame.to_csv(output_dir / "v2_base_frame.csv", index=False)
        try:
            result = pipeline.run_v2_next_session_prediction(
                v2_base_frame,
                forecaster_config,
                experiment_name,
                basket_mode=args.basket_mode,
                book_mode=args.book_mode,
                tail_fraction=args.tail_fraction,
            )
        except ValueError as exc:
            print()
            print("Live prediction blocked")
            print(str(exc))
            return
    else:
        labeled_frame = pipeline.prepare_labeled_frame(tickers=tickers, start=start, end=end)
        labeled_frame.to_csv(output_dir / "labeled_frame.csv", index=False)
        result = pipeline.run_next_session_prediction(labeled_frame, forecaster_config, experiment_name)

    prediction_dir = prediction_root / experiment_name
    ZeroShotResearchPipeline.write_live_prediction(prediction_dir, result)

    reconcile_status = "pending"
    reconcile_message = ""
    prediction_sessions: list[pd.Timestamp] = []
    if not result.prediction_summary.empty and "prediction_session" in result.prediction_summary.columns:
        prediction_sessions = (
            pd.to_datetime(result.prediction_summary["prediction_session"], utc=True, errors="coerce")
            .dropna()
            .unique()
            .tolist()
        )
    elif not result.book.frame.empty and "prediction_session" in result.book.frame.columns:
        prediction_sessions = (
            pd.to_datetime(result.book.frame["prediction_session"], utc=True, errors="coerce")
            .dropna()
            .unique()
            .tolist()
        )

    can_reconcile, reconcile_message = pipeline.reconciliation_readiness(
        prediction_sessions,
        market_timezone=pipeline.research_config.market_timezone,
    )
    if can_reconcile:
        try:
            pipeline.reconcile_live_prediction(
                prediction_dir=prediction_dir,
                ledger_dir=ledger_dir,
                execution_interval=args.execution_interval,
                execution_mode=args.execution_mode,
            )
            reconcile_status = "completed"
            reconcile_message = ""
        except Exception as exc:  # pragma: no cover - live data availability can vary by date
            reconcile_status = "pending"
            reconcile_message = str(exc)

    export_chronos_package(
        prediction_dir=prediction_dir,
        analysis_dir=Path(args.analysis_dir) if args.analysis_dir else None,
        ledger_dir=ledger_dir if ledger_dir.exists() else None,
        output_dir=report_dir,
        report_mode="live",
        report_context={
            "prediction_dir": str(prediction_dir),
            "analysis_dir": str(Path(args.analysis_dir)) if args.analysis_dir else "",
            "ledger_dir": str(ledger_dir),
            "reconciliation_status": reconcile_status,
            "reconciliation_message": reconcile_message,
        },
    )
    _print_live_summary(prediction_dir, report_dir, reconcile_status, reconcile_message)
    _print_terminal_report(report_dir)


def run_backtest_mode(args: argparse.Namespace) -> None:
    start, end = _default_backtest_window()
    if args.start:
        start = args.start
    if args.end:
        end = args.end

    base_tickers = _parse_tickers(args.tickers, args.tickers_file)
    tickers = _resolve_v2_tickers(base_tickers, args.architecture, args.basket_mode)
    if args.product_mode == "winner_picker" and args.architecture != "v2_flagship_basket_raw":
        print("winner_picker is supported only with v2_flagship_basket_raw.")
        return
    pipeline, context_length = _build_pipeline(args, tickers)
    forecaster_config = _build_forecaster_config(args, context_length)
    experiment_name = _build_experiment_name(args, context_length)
    output_dir = Path(args.output_dir) if args.output_dir else _default_output_dir("backtest")
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.architecture == "v2_flagship_basket_raw":
        labeled_frame, v2_base_frame = pipeline.prepare_market_frames(tickers=tickers, start=start, end=end)
        labeled_frame.to_csv(output_dir / "labeled_frame.csv", index=False)
        v2_base_frame.to_csv(output_dir / "v2_base_frame.csv", index=False)
        result = pipeline.run_v2_experiment(
            v2_base_frame,
            forecaster_config,
            experiment_name,
            basket_mode=args.basket_mode,
            book_mode=args.book_mode,
            tail_fraction=args.tail_fraction,
        )
    else:
        labeled_frame = pipeline.prepare_labeled_frame(tickers=tickers, start=start, end=end)
        labeled_frame.to_csv(output_dir / "labeled_frame.csv", index=False)
        result = pipeline.run_experiment(labeled_frame, forecaster_config, experiment_name)

    experiment_dir = output_dir / experiment_name
    ZeroShotResearchPipeline.write_result(experiment_dir, result)

    analysis_dir: Path | None = None
    if args.architecture == "v2_flagship_basket_raw" and args.product_mode == "ranker" and not args.skip_analysis:
        analysis_dir = output_dir / f"{experiment_name}_analysis"
        pipeline.analyze_signal_run(
            run_dir=output_dir,
            experiment=experiment_name,
            output_dir=analysis_dir,
            score_column=args.score_column,
            tail_fractions=tuple(args.analysis_tail_fractions),
            execution_interval=args.execution_interval,
            execution_mode=args.execution_mode,
            default_borrow_fee_annual_bps=args.borrow_fee_annual_bps,
            borrow_profile_path=Path(args.borrow_profile_csv) if args.borrow_profile_csv else None,
            unavailable_tickers=_normalize_optional_tickers(args.borrow_unavailable_tickers),
        )
    _print_backtest_summary(experiment_dir, analysis_dir)


def run_report_mode(args: argparse.Namespace) -> None:
    prediction_dir = Path(args.prediction_dir)
    output_dir = Path(args.output_dir) if args.output_dir else _default_output_dir("report")
    export_chronos_package(
        prediction_dir=prediction_dir,
        analysis_dir=Path(args.analysis_dir) if args.analysis_dir else None,
        ledger_dir=Path(args.ledger_dir) if args.ledger_dir else None,
        output_dir=output_dir,
        report_mode="report_only",
        report_context={
            "prediction_dir": str(prediction_dir),
            "analysis_dir": str(Path(args.analysis_dir)) if args.analysis_dir else "",
            "ledger_dir": str(Path(args.ledger_dir)) if args.ledger_dir else "",
            "reconciliation_status": "not_run",
            "reconciliation_message": "",
        },
    )
    print()
    print("Report package created")
    print(f"  Prediction dir: {prediction_dir}")
    print(f"  Report file:    {output_dir / 'paper_report.html'}")
    _print_terminal_report(output_dir)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Main Chronos workstation. Use one file for live prediction, backtesting, or report generation."
    )
    parser.add_argument(
        "--mode",
        required=True,
        choices=("live", "backtest", "report"),
        help="Workstation mode.",
    )
    parser.add_argument("--tickers", help="Comma-separated tickers.")
    parser.add_argument("--tickers-file", help="Path to a newline-delimited ticker file.")
    parser.add_argument("--start", help="Inclusive start date in YYYY-MM-DD format.")
    parser.add_argument("--end", help="Exclusive end date in YYYY-MM-DD format.")
    parser.add_argument("--output-dir", help="Directory where workstation outputs will be written.")
    parser.add_argument("--prediction-dir", help="Existing predict-next experiment directory for report mode.")
    parser.add_argument("--analysis-dir", help="Optional frozen analysis directory to include in the report.")
    parser.add_argument("--ledger-dir", help="Optional cumulative live scoreboard directory.")
    parser.add_argument("--model-id", default="amazon/chronos-2")
    parser.add_argument("--prediction-length", type=int, default=7)
    parser.add_argument("--context-length", type=int)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--n-long", type=int, default=5)
    parser.add_argument("--n-short", type=int, default=5)
    parser.add_argument("--benchmark", default="SPY")
    parser.add_argument("--cross-learning", action="store_true")
    parser.add_argument("--no-covariates", action="store_true")
    parser.add_argument(
        "--product-mode",
        default=DEFAULT_PRODUCT_MODE,
        choices=("ranker", "winner_picker"),
    )
    parser.add_argument(
        "--confirmation-mode",
        default=DEFAULT_WINNER_CONFIRMATION_MODE,
        choices=DEFAULT_WINNER_CONFIRMATION_MODES,
    )
    parser.add_argument(
        "--architecture",
        default="v2_flagship_basket_raw",
        choices=("v1_residual", "v2_flagship_basket_raw"),
    )
    parser.add_argument(
        "--basket-mode",
        default="expanded40",
        choices=("full", "growth_spike", "expanded40", "expanded60"),
    )
    parser.add_argument(
        "--beta-mode",
        default=DEFAULT_V2_BETA_MODE,
        choices=("unit_beta_1.0", "pooled_beta_140", "rolling_session_beta_20d"),
    )
    parser.add_argument(
        "--book-mode",
        default=DEFAULT_V2_BOOK_MODE,
        choices=("tail_fraction", "fixed_count"),
    )
    parser.add_argument("--tail-fraction", type=float, default=DEFAULT_V2_TAIL_FRACTION)
    parser.add_argument("--execution-interval", default="5m")
    parser.add_argument("--execution-mode", default="strict", choices=("strict",))
    parser.add_argument("--skip-analysis", action="store_true", help="Backtest mode only: skip analyze-signal.")
    parser.add_argument("--score-column", default="score_z", help="Backtest analysis score column.")
    parser.add_argument(
        "--analysis-tail-fractions",
        default="0.10,0.15,0.20",
        help="Comma-separated tail fractions used by analyze-signal in backtest mode.",
    )
    parser.add_argument("--borrow-fee-annual-bps", type=float, default=DEFAULT_SHORT_BORROW_FEE_ANNUAL_BPS)
    parser.add_argument("--borrow-profile-csv", help="Optional CSV with ticker, borrow_fee_annual_bps, short_available.")
    parser.add_argument("--borrow-unavailable-tickers", help="Optional comma-separated short tickers to exclude in borrow overlay.")
    return parser


def main() -> None:
    if len(sys.argv) == 1:
        try:
            _run_interactive_menu()
        except KeyboardInterrupt:
            print()
            print("Chronos Workstation closed.")
        return

    parser = build_parser()
    args = parser.parse_args()
    args.analysis_tail_fractions = _normalize_analysis_tail_fractions(args.analysis_tail_fractions)

    if args.mode == "live":
        run_live_mode(args)
        return
    if args.mode == "backtest":
        run_backtest_mode(args)
        return
    if args.mode == "report":
        if not args.prediction_dir:
            parser.error("--prediction-dir is required when --mode report")
        run_report_mode(args)
        return


if __name__ == "__main__":
    main()
