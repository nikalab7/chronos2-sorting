# Chronos Equity Prototype

This repository contains a zero-shot Chronos-2 research prototype for US equities using Yahoo Finance intraday data as a pipeline-validation source.

## What It Does

- Downloads regular-hours intraday bars from Yahoo Finance
- Normalizes each session into 7 slots:
  - 09:30
  - 10:30
  - 11:30
  - 12:30
  - 13:30
  - 14:30
  - 15:30
- Builds slot-level residual return labels versus `SPY`
- Creates official Chronos-2 list-of-dicts forecasting tasks
- Runs zero-shot forecasts with the flagship `amazon/chronos-2` model on CPU
- Builds an equal-weight dollar-neutral long-short book
- Reports gross and net results across cost stress tests
- Benchmarks against simple non-ML baselines

## Install

```bash
pip install -e .
```

If you only want to run the unit tests for the non-model math, the codebase works without `chronos-forecasting` being installed. The Chronos import is lazy and only required for the forecasting step.

## Quick Start

```bash
python -m chronos_equity_prototype.cli run ^
  --start 2026-02-15 ^
  --end 2026-04-16 ^
  --architecture v2_flagship_basket_raw ^
  --output-dir outputs/default_run
```

The project now defaults to the 120M flagship `amazon/chronos-2` model with a CPU-safer default batch size of `8`.
For V2, the runtime defaults are:

- `context_length = 224`
- `beta_mode = unit_beta_1.0`
- `book_mode = tail_fraction`
- `tail_fraction = 0.10`
- `ranking_score = score_z`

## One Main Workstation File

If you want a single entrypoint instead of switching between multiple CLI commands, use the repo-root [main.py](</c:/Users/user/Desktop/New folder/main.py>).

If you simply run:

```bash
python main.py
```

it now opens an interactive workstation menu with guided options for:

- live prediction with paper report
- backtest with analysis
- building a report from an existing prediction folder

Live prediction plus optional live reconciliation plus visual paper report:

```bash
python main.py ^
  --mode live ^
  --architecture v2_flagship_basket_raw ^
  --basket-mode expanded40 ^
  --analysis-dir outputs/fresh_locked_v2_expanded40_analysis
```

Historical backtest plus optional signal/execution analysis:

```bash
python main.py ^
  --mode backtest ^
  --architecture v2_flagship_basket_raw ^
  --basket-mode expanded40
```

Build a visual report package from existing prediction files:

```bash
python main.py ^
  --mode report ^
  --prediction-dir outputs/next_session_expanded40_locked_single_session/v2_flagship_basket_raw_expanded40_amazon_chronos_2_ctx224_cross0_cov1 ^
  --analysis-dir outputs/fresh_locked_v2_expanded40_analysis ^
  --ledger-dir outputs/live_scoreboard_expanded40
```

The main workstation prints the key results in the terminal and writes a visual `paper_report.html` package with charts, current prediction tables, confidence, ticker-level paths, and optional live/backtest reference boards.

## Predict The Next Session

```bash
python -m chronos_equity_prototype.cli predict-next ^
  --architecture v2_flagship_basket_raw ^
  --output-dir outputs/next_session
```

This command uses the latest complete regular-hours session in your Yahoo window as the signal date, forecasts the next US trading session's 7 slots, writes `forecasts.json`, `book.csv`, and `prediction_summary.csv`, and prints the long/short book in the terminal.

## Analyze A Frozen V2 Run

```bash
python -m chronos_equity_prototype.cli analyze-signal ^
  --run-dir outputs/v2_matrix_run ^
  --experiment v2_flagship_basket_raw_full_amazon_chronos_2_ctx224_cross0_cov1 ^
  --output-dir outputs/analysis_v2_ctx224
```

This command does not rerun Chronos-2. It loads a frozen V2 experiment, rebuilds the default tail-only book, writes signal-quality diagnostics, and runs a strict 5-minute near-open execution overlay with:

- `exact_open`
- `first_5m_avg`
- `first_15m_avg`

## Reconcile A Saved Live Prediction

```bash
python -m chronos_equity_prototype.cli reconcile-live ^
  --prediction-dir outputs/v2_next_ctx256/v2_flagship_basket_raw_full_amazon_chronos_2_ctx256_cross0_cov1 ^
  --ledger-dir outputs/live_reconciliation_ctx256
```

This command does not rerun Chronos-2. It loads a saved `predict-next` experiment, fetches the realized next session, appends a cumulative paper-trading ledger, and writes:

- `live_reconciled_forecasts.csv`
- `live_scoreboard_positions.csv`
- `live_scoreboard_daily.csv`
- `live_rank_ic.csv`
- `live_metrics_summary.csv`
- `live_execution_sensitivity_comparison.csv`

## Build A Dated Chronos Report Package

```bash
python -m chronos_equity_prototype.cli chronos-report ^
  --prediction-dir outputs/next_session_expanded40_locked/v2_flagship_basket_raw_expanded40_amazon_chronos_2_ctx224_cross0_cov1 ^
  --analysis-dir outputs/fresh_locked_v2_expanded40_analysis ^
  --output-dir outputs/chronos_2026-04-17
```

This command creates a human-readable package with:

- `prediction_board.csv`
- `prediction_slot_board.csv`
- `prediction_summary_board.csv`
- `backtest_overview.csv`
- `backtest_daily_board.csv`
- `backtest_position_board.csv`
- `backtest_ticker_board.csv`
- `README.txt`

## Test

```bash
python -m unittest discover -s tests -v
```
