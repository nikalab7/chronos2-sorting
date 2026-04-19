# Chronos Equity Prototype System Architecture

This document explains the full architecture of the current Chronos equity prototype:

- what data enters the system
- how sessions are normalized
- how V1 and V2 forecasting paths differ
- how scores and books are built
- how backtest and live reconciliation work
- where the most important audit risks and possible improvements live

The goal is to make the whole pipeline understandable enough that you can review it mathematically and operationally without reverse-engineering every file from scratch.

## 1. System Goal

The current project is a zero-shot equity forecasting and ranking system built around Chronos-2.

The production-like research target is:

- use US regular-hours intraday data
- represent each session as 7 ordered slots
- forecast the next session after the close
- build a dollar-neutral long-short rank book
- analyze both historical performance and live paper-trading behavior

There are two architecture paths in the repo:

- `v1_residual`
- `v2_flagship_basket_raw`

The current locked baseline is V2, not V1.

## 2. Top-Level Flow

At a high level the pipeline is:

1. Fetch intraday bars from Yahoo Finance.
2. Convert bars into regular-session slot geometry.
3. Build either:
   - V1 residual labels, or
   - V2 raw-return basket features.
4. Build Chronos tasks / basket prediction inputs.
5. Run Chronos-2 zero-shot inference on CPU.
6. Convert forecasts into alpha scores.
7. Build a tail-ranked long-short book.
8. Run backtest or save live prediction outputs.
9. If the predicted day has finished, reconcile realized returns and execution overlays.
10. Generate CSV diagnostics and human-readable reports.

Main orchestrator:

- [src/chronos_equity_prototype/pipeline.py](src/chronos_equity_prototype/pipeline.py)

Interactive workstation entrypoint:

- [main.py](main.py)

CLI entrypoint:

- [src/chronos_equity_prototype/cli.py](src/chronos_equity_prototype/cli.py)

## 3. Main Files And Responsibilities

### Core configuration

- [src/chronos_equity_prototype/config.py](src/chronos_equity_prototype/config.py)
  - default model id
  - default context length
  - default prediction length
  - default V2 beta mode
  - default book mode
  - cost assumptions

### Data fetch and normalization

- [src/chronos_equity_prototype/data.py](src/chronos_equity_prototype/data.py)
  - `YahooIntradayProvider`
  - `SessionNormalizer`

### Labels

- [src/chronos_equity_prototype/labels.py](src/chronos_equity_prototype/labels.py)
  - V1 residual target construction

### V1 task builder

- [src/chronos_equity_prototype/tasks.py](src/chronos_equity_prototype/tasks.py)
  - per-stock Chronos tasks
  - US trading-day calendar logic

### Forecast model interface

- [src/chronos_equity_prototype/forecasting.py](src/chronos_equity_prototype/forecasting.py)
  - lazy Chronos-2 pipeline loading
  - V1 list-of-dicts prediction
  - V2 `predict_df` prediction

### V2 basket architecture

- [src/chronos_equity_prototype/v2_basket.py](src/chronos_equity_prototype/v2_basket.py)
  - basket definitions
  - base-frame feature engineering
  - missing-data policy
  - multivariate basket input building
  - alpha scoring and flattening

### Portfolio construction

- [src/chronos_equity_prototype/portfolio.py](src/chronos_equity_prototype/portfolio.py)
  - fixed-count and tail-fraction book construction

### Backtest engine

- [src/chronos_equity_prototype/backtest.py](src/chronos_equity_prototype/backtest.py)
  - gross and net portfolio returns
  - Sharpe and win-rate reporting

### Baselines

- [src/chronos_equity_prototype/baselines.py](src/chronos_equity_prototype/baselines.py)
  - simple non-ML comparison signals

### Diagnostics and live reconciliation

- [src/chronos_equity_prototype/diagnostics.py](src/chronos_equity_prototype/diagnostics.py)
  - rank IC
  - tail studies
  - basket and side decomposition
  - near-open execution overlays
  - live reconciliation outputs
  - borrow overlay

### Report packaging

- [src/chronos_equity_prototype/reporting.py](src/chronos_equity_prototype/reporting.py)
  - prediction board
  - slot board
  - live/backtest boards
  - paper report HTML and charts

## 4. Data Layer

### 4.1 Source

The current source is Yahoo Finance intraday data through `yfinance`.

Important limitations:

- intraday history is short
- some days or bars can be incomplete
- quality is good enough for a prototype, not strong enough for production evidence

### 4.2 Regular-hours session geometry

The system maps each day into exactly 7 regular-hours slots:

1. `09:30`
2. `10:30`
3. `11:30`
4. `12:30`
5. `13:30`
6. `14:30`
7. `15:30`

Those are implemented in:

- [src/chronos_equity_prototype/data.py](src/chronos_equity_prototype/data.py)
- [src/chronos_equity_prototype/v2_basket.py](src/chronos_equity_prototype/v2_basket.py)

The final slot is the close slot representing the last half-hour of the regular session.

### 4.3 Timezone handling

All session logic is anchored to:

- `America/New_York`

Raw bars are converted into New York time before:

- session assignment
- slot assignment
- live prediction freshness checks
- reconciliation readiness checks

This is important because the system must reason about US market days, not local machine time.

## 5. Two Internal Data Products

When V2 is used, the pipeline intentionally creates two related but different frames:

### 5.1 `labeled_frame`

Built from complete sessions only.

Used for:

- V1 forecasting
- realized-outcome attachment
- historical backtest math
- live reconciliation after the predicted day has finished

### 5.2 `v2_base_frame`

Built from normalized data with incomplete sessions allowed.

Used for:

- V2 basket preparation
- latest common signal-session selection
- live prediction-time freshness checks

This split is important:

- `labeled_frame` is the clean realized-data frame
- `v2_base_frame` is the live/raw working frame

## 6. V1 Architecture

V1 is the older residual-first design.

### 6.1 V1 target

For stock `i`, session `t`, slot `h`:

`slot_log_return(i,t,h) = log(C_i / O_i)`

`benchmark_slot_return(t,h) = log(C_SPY / O_SPY)`

`y_slot(i,t,h) = slot_log_return(i,t,h) - benchmark_slot_return(t,h)`

This is implemented in:

- [src/chronos_equity_prototype/labels.py](src/chronos_equity_prototype/labels.py)

### 6.2 V1 covariates

Past-only covariates:

- `log_volume`
- `high_low_range`
- `rolling_slot_vol_20`
- `spy_same_slot_return`

Known-future covariates:

- `slot_id`
- `weekday`
- `month_end_flag`

### 6.3 V1 task shape

Each stock becomes one Chronos task.

History:

- last `context_length` slot-level residual returns

Future horizon:

- next `prediction_length = 7` slots

### 6.4 V1 model output

The model returns quantile paths:

- `q10_path`
- `q50_path`
- `q90_path`

V1 score:

- `score_q50 = sum(q50_path)`

### 6.5 V1 role today

V1 is still useful as:

- a control
- a sanity-check architecture
- a simpler fallback path

But it is not the current main research baseline.

## 7. V2 Architecture

V2 is the current flagship-first design.

It does not forecast pre-built residual alpha directly.
Instead it forecasts raw slot returns jointly for stock baskets plus `SPY`, then derives alpha after the forecast.

### 7.1 Why V2 exists

The idea behind V2 is:

- let Chronos-2 see more market structure
- let the model forecast stock and benchmark together
- derive relative alpha after forecasting rather than before forecasting

This better matches Chronos-2’s multivariate and covariate-informed strengths.

### 7.2 Basket definitions

Current basket modes are defined in:

- [src/chronos_equity_prototype/v2_basket.py](src/chronos_equity_prototype/v2_basket.py)

Main modes:

- `full`
- `growth_spike`
- `expanded40`
- `expanded60`

The current locked workflow uses:

- `expanded40`

That means 40 target stocks total split across:

- `growth`
- `defensive_value`

Each basket also includes `SPY`.

### 7.3 V2 raw target

For each ticker and slot:

`raw_slot_return = log(close / open)`

Unlike V1, this is not residualized before model inference.

### 7.4 V2 engineered features

Asset-level past-only features:

- `log_volume`
- `high_low_range`
- `rolling_slot_vol_20`
- `overnight_gap`
- `prior_session_return`

Basket-level past-only features:

- `basket_mean_return_ex_spy`
- `basket_dispersion`
- `basket_positive_breadth`
- `basket_realized_vol`

Known-future features:

- `slot_id`
- `weekday`
- `month_end_flag`

### 7.5 V2 missing-data rules

The basket frame is cleaned with explicit rules.

If `SPY` is incomplete:

- drop the basket-date

If a stock misses the whole session:

- drop the basket-date

If a stock misses multiple slots:

- drop the basket-date

If a stock misses exactly one isolated interior slot:

- midpoint-impute and log the event

This policy is strict on purpose because V2 depends on basket consistency.

### 7.6 V2 latest-session selection

For live mode, V2 does not blindly use the newest session on any one ticker.
It finds the latest common signal session across all prepared baskets that:

- survived missing-data checks
- has enough history for the chosen context length

This is the mechanism that decides whether today is usable as the live signal session.

### 7.7 V2 Chronos input representation

Chronos-2 receives:

- a multivariate `history_df`
- a `future_df` with known-future covariates
- one `item_id` per basket-date input

Targets are columns for all basket members including `SPY`.

Internally the code creates synthetic hourly timestamps.
Those timestamps are just sequence scaffolding for the model input table.
The actual economic meaning comes from the ordered slot structure and covariates.

## 8. Forecasting Layer

### 8.1 Model loading

The forecasting layer is in:

- [src/chronos_equity_prototype/forecasting.py](src/chronos_equity_prototype/forecasting.py)

Chronos is lazy-loaded via:

- `Chronos2Pipeline.from_pretrained(model_id, device_map="cpu")`

### 8.2 V1 inference path

Uses:

- `predict_quantiles(inputs=..., ...)`

### 8.3 V2 inference path

Uses:

- `predict_df(df=..., future_df=..., target=..., ...)`

### 8.4 Current locked baseline config

Current winning baseline:

- `architecture = v2_flagship_basket_raw`
- `model_id = amazon/chronos-2`
- `context_length = 224`
- `prediction_length = 7`
- `cross_learning = False`
- `use_covariates = True`
- `beta_mode = unit_beta_1.0`
- `ranking_score = score_z`
- `book_mode = tail_fraction`
- `tail_fraction = 0.10`

## 9. Alpha Construction And Scoring

### 9.1 Beta mode

Two beta modes currently exist:

- `unit_beta_1.0`
- `pooled_beta_140`

Current default:

- `unit_beta_1.0`

That means every stock is adjusted against `SPY` one-for-one.

### 9.2 V2 forecast alpha

For stock `i` and forecast step `h`:

`alpha_hat_i[h] = raw_hat_i[h] - beta_i * raw_hat_spy[h]`

With the locked baseline:

`beta_i = 1.0`

### 9.3 Main score layers

The flattened V2 output contains:

- `score_plain`
- `score_risk`
- `score_z`

Definitions:

`score_plain = sum_h alpha_hat_q50[h]`

`alpha_uncertainty = sum_h ((q90_stock[h] - q10_stock[h]) + |beta_i| * (q90_spy[h] - q10_spy[h]))`

`score_risk = score_plain / (eps + alpha_uncertainty)`

`score_z = zscore(score_plain within basket_id and prediction_session)`

### 9.4 Important compatibility detail

For legacy compatibility:

- legacy frozen V2 runs may still contain `score_q50` as an alias for `score_z`

Newly written V2 CSV and JSON outputs do not expose `score_q50`; older frozen runs may still contain it so the older ranking pipeline can still work.

That means:

- new research and reporting should read `score_plain`, `score_risk`, `score_z`, `ranking_score`, and `ranking_score_column`

- `score_q50` in those old runs no longer literally means "sum of q50 path"

This is mathematically okay as an implementation shortcut, but it is a semantic trap when reading CSVs.

## 10. Portfolio Construction

Portfolio logic is in:

- [src/chronos_equity_prototype/portfolio.py](src/chronos_equity_prototype/portfolio.py)

### 10.1 Two book modes

- `fixed_count`
- `tail_fraction`

V1 and baselines typically use:

- `fixed_count`

V2 default uses:

- `tail_fraction`

### 10.2 Tail-fraction math

If the universe has `N` names and `tail_fraction = f`, then:

- number of longs = `ceil(N * f)`
- number of shorts = `ceil(N * f)`

Weights are:

- long weight per name = `+0.5 / n_each`
- short weight per name = `-0.5 / n_each`

So the book is:

- equal-weight
- 50/50 long-short
- dollar-neutral by construction

For `expanded40` with `tail_fraction = 0.10`:

- `N = 40`
- longs = `4`
- shorts = `4`

That is why the live book currently has 8 names total.

## 11. Backtest Math

Backtest logic is in:

- [src/chronos_equity_prototype/backtest.py](src/chronos_equity_prototype/backtest.py)

For each booked position:

`gross_position_return = weight * realized_day_return`

For each day:

`gross_return = sum(gross_position_return)`

`gross_exposure = sum(abs(weight))`

For cost level `c` in bps:

`net_return = gross_return - gross_exposure * c / 10000`

Daily Sharpe is annualized as:

`sqrt(252) * mean(daily_returns) / std(daily_returns)`

## 12. Diagnostics Layer

Diagnostics are built from frozen forecasts and realized data in:

- [src/chronos_equity_prototype/diagnostics.py](src/chronos_equity_prototype/diagnostics.py)

Major outputs include:

- score behavior comparison
- rank IC
- tail comparison
- execution sensitivity
- basket contribution
- side contribution
- ticker attribution
- OOS concentration
- side isolation
- borrow overlay

### 12.1 Rank IC

Rank IC is computed cross-sectionally by day:

- rank forecast scores
- rank realized returns
- compute correlation of the ranks

This is implemented in:

- `compute_rank_ic_for_score()`

### 12.2 Tail analysis

The analysis layer can rebuild books for multiple tail sizes without rerunning the model.
That is how you compare:

- 10%
- 15%
- 20%

tails on the same frozen forecasts.

### 12.3 Basket and side decomposition

The diagnostics layer splits results into:

- combined
- growth only
- defensive only
- long only
- short only

This is very important because total Sharpe can hide weak sub-regimes.

## 13. Near-Open Execution Overlay

Execution-overlay logic is also in:

- [src/chronos_equity_prototype/diagnostics.py](src/chronos_equity_prototype/diagnostics.py)

It fetches 5-minute bars and constructs three entry proxies.

### 13.1 Entry formulas

`exact_open`

- open of the `09:30` 5-minute bar

`first_5m_avg`

- OHLC4 of the first 5-minute bar
- `(O + H + L + C) / 4`

`first_15m_avg`

- volume-weighted average of OHLC4 across the `09:30`, `09:35`, and `09:40` 5-minute bars

Exit:

- close of the final regular-hours `15:55` 5-minute bar

### 13.2 Execution alpha

For each ticker:

`realized_execution_alpha = log(stock_exit / stock_entry) - log(spy_exit / spy_entry)`

In the current execution overlay:

- unit beta is used

### 13.3 Strict mode

Strict execution benchmark mode drops a whole prediction day if:

- `SPY` is missing any required opening or closing 5-minute bars
- any booked constituent is missing required opening or closing 5-minute bars

This keeps the execution comparison conservative.

## 14. Live Prediction Workflow

Live mode is handled in:

- [main.py](main.py)
- [src/chronos_equity_prototype/pipeline.py](src/chronos_equity_prototype/pipeline.py)

### 14.1 Correct live logic

The intended logic is:

1. Define “now” in New York market time.
2. Find the latest fully completed common session across baskets.
3. Set `prediction_session` to the next US trading day.
4. Block if `prediction_session` is not strictly after the current New York market day.
5. Save prediction files.
6. Do not fetch realized 5-minute execution data yet.

### 14.2 Freshness and readiness

The system now separates:

- prediction readiness
- reconciliation readiness

Prediction readiness means:

- the target day is strictly after the current market day

Reconciliation readiness means:

- the predicted session already exists
- and the regular session is fully complete

This separation prevents future execution-data fetches by mistake.

## 15. Live Reconciliation Workflow

Live reconciliation is in:

- [src/chronos_equity_prototype/pipeline.py](src/chronos_equity_prototype/pipeline.py)
- [src/chronos_equity_prototype/diagnostics.py](src/chronos_equity_prototype/diagnostics.py)

It happens after the predicted session finishes.

Steps:

1. Load saved `forecasts.json`, `book.csv`, and `prediction_summary.csv`.
2. Check that the prediction session is eligible for reconciliation.
3. Fetch realized 60-minute bars to rebuild true next-session returns.
4. Fetch realized 5-minute bars for execution overlay.
5. Attach realized slot paths and realized day returns.
6. Recompute live book results.
7. Append to the cumulative live ledger.

Ledger files include:

- `prediction_summary_history.csv`
- `live_reconciled_forecasts.csv`
- `live_scoreboard_positions.csv`
- `live_scoreboard_daily.csv`
- `live_execution_position_returns.csv`
- `live_execution_daily_returns.csv`

## 16. Reporting Layer

The reporting layer builds human-readable outputs from saved prediction and analysis files.

Main file:

- [src/chronos_equity_prototype/reporting.py](src/chronos_equity_prototype/reporting.py)

Main products:

- `prediction_board.csv`
- `slot_board.csv`
- `paper_summary.csv`
- `paper_report.html`

The report also builds heuristic fields like:

- `confidence_proxy`
- `prediction_rank_desc`
- `confidence_rank_desc`

Important note:

- `confidence_proxy` is not a calibrated probability
- it is only a derived ratio of score magnitude to alpha uncertainty

## 17. Known Weak Spots Worth Auditing

These are the most important architecture and math risks currently worth reviewing.

### 17.1 Baseline duplication

In [src/chronos_equity_prototype/baselines.py](src/chronos_equity_prototype/baselines.py), these two baselines are effectively the same type of daily residual-sum rank:

- `last_same_slot_residual_return`
- `one_day_residual_momentum_rank` (now removed from the default baseline suite)

That weakens the acceptance test because the baseline set looks more diverse than it really is.

### 17.2 Misleading feature naming

`rolling_slot_vol_20` is a rolling 20-slot volatility, not 20 daily sessions.

That definition is mathematically fine for the current prototype, but it should not be confused with a 20-session daily volatility feature.

### 17.3 Legacy V2 `score_q50` naming trap

In older V2 frozen runs:

- `score_q50` is not literally the sum of median alpha anymore
- it is the normalized execution score `score_z`

This should always be remembered when reading outputs or comparing regimes.

### 17.4 Basket-common session dependency

V2 live mode requires a latest common usable signal session across baskets.
That means a single broken latest-day ticker can roll the whole live signal back by one session.

This is operationally clean but can be frustrating in real-time use.

### 17.5 Yahoo quality pressure

Because the live signal depends on intraday Yahoo bars:

- missing last-slot bars
- stale updates
- short history

can affect:

- usable signal dates
- basket survival
- apparent live readiness

### 17.6 Confidence versus probability

The current report exposes strong "confidence-like" outputs, but they are not model-calibrated probabilities.
They should be treated as heuristic ranking aids, not literal forecast confidence.

## 18. Current Locked Baseline

The current working baseline is:

- architecture: `v2_flagship_basket_raw`
- model: `amazon/chronos-2`
- context length: `224`
- prediction length: `7`
- cross learning: `False`
- covariates: `True`
- beta mode: `unit_beta_1.0`
- ranking score: `score_z`
- book mode: `tail_fraction`
- tail fraction: `0.10`
- universe mode: currently `expanded40`

This is the configuration the rest of the current trading-layer analysis is built around.

## 19. Practical Review Checklist

If you want to audit the system for bugs or accuracy opportunities, this is the best order:

1. Confirm the target definition you really want:
   - V1 residual target
   - or V2 raw-return then post-forecast alpha

2. Confirm session geometry:
   - 7 slots exactly
   - no pre/post leakage
   - no timezone drift

3. Confirm latest-signal selection in live mode:
   - based on New York market time
   - based on latest fully completed common session

4. Confirm score semantics:
   - `score_plain` is raw alpha strength
   - `score_z` is executable normalized score
   - legacy `score_q50` in old V2 runs is an alias

5. Confirm the evaluation target:
   - residual next-day alpha
   - or raw next-day stock return
   - and whether execution overlay should remain unit-beta

6. Audit baseline diversity:
   - remove duplicates
   - add truly different simple comparisons

7. Audit whether slot-based versus session-based volatility features are more appropriate.

## 20. Bottom Line

The current system is best understood as:

- a strict intraday market-data pipeline
- feeding a multivariate Chronos-2 raw-return forecaster
- then converting forecasts into unit-beta market-relative alpha
- then normalizing within basket
- then extracting a tail-only long-short book
- then measuring both historical and near-open executable behavior

The architecture is coherent.
The biggest remaining review opportunities are not basic wiring errors.
They are:

- baseline honesty
- feature-definition precision
- score semantics clarity
- live robustness under imperfect Yahoo data
