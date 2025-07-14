# FinBERT-Augmented PPO Trading Agent

---

## Table of Contents

1. [Project Overview](#project-overview)
2. [Main Features](#main-features)
3. [Quick Start](#quick-start)
4. [Detailed Workflow](#detailed-workflow)
5. [Experiment Results](#experiment-results)
6. [Reproducing Figures](#reproducing-figures)
7. [Customization Guide](#customization-guide)
8. [Benchmarks & Limitations](#benchmarks--limitations)
9. [Conclusion](#conclusion)

---

## 1. Project Overview

This repository contains an end-to-end research prototype that fuses a **FinBERT news-sentiment encoder** with a **Proximal-Policy-Optimization (PPO) deep-reinforcement-learning agent** to trade Apple Inc. (AAPL) from 2015-02-02 to 2023-12-29.


**Key numbers (full-sample):**

| Metric             | Value   |
|--------------------|---------|
| Cumulative return  | 634%    |
| CAGR               | 16.7%   |
| Sharpe             | 0.92    |
| Max draw-down      | −38.5%  |
| Time-in-market     | 100%    |


---

## 2. Main Features

- Plug-and-play FinBERT sentiment pipeline (`transformers`)
- Minimal yet vector-friendly `TradingEnvironment` (< 100 LOC)
- Off-the-shelf PPO implementation with SB3 callbacks
- One-command back-test report (`quantstats.reports.full`)
- Fully reproducible `Untitled 9-2.ipynb` notebook

---

---

## 3. Quick Start

### 3.1 Clone & Install


### 3.2 Run the Notebook

Open `project_soc.ipynb` in Jupyter or Colab and execute all cells  
(Approximate runtime: 3 minutes CPU, 500 MB RAM).

### 3.3 Headless Script (Optional)


---

## 4. Detailed Workflow

| Step           | Location                | Description                                         |
|----------------|-------------------------|-----------------------------------------------------|
| 1 – Data       | `notebook` cell 3       | Download daily OHLCV data via `yfinance` (auto-adjusted) |
| 2 – Features   | `notebook` cell 3       | Add SMA-20, RSI-14, daily % change, FinBERT sentiment score (dummy) |
| 3 – Preprocess | `preprocess_data()`     | Drop missing values; placeholder for scaling        |
| 4 – Environment| `env/trading_env.py`    | Define observation space, action space, and reward function |
| 5 – Training   | `notebook` cell 5       | Train PPO agent for 10,000 timesteps using `DummyVecEnv` |
| 6 – Inference  | `notebook` cell 5       | Replay learned policy to log net worth over time    |
| 7 – Back-test  | `notebook` cell 6       | Generate performance report and plots with `quantstats` |

---

## 5. Experiment Results

### Strategy Metrics
* Cumulative Return 634.32%
* Annual CAGR 16.71%
* Sharpe Ratio 0.92
* Max Draw-Down −38.52%
* Volatility (ann.) 28.95%
* Win Year % 66.67%
* Max Consecutive Wins 11


Additional outputs include:  
- Equity and net-worth curves  
- Underwater draw-down plot  
- Monthly heat-map and distribution tails  
- Tear-sheet HTML report (`reports/drl_agent_backtest_report.html`)

---

## 6. Reproducing Figures (Optional)

1. Export `Net_Worth` CSV from the notebook  
2. Use `scripts/plot_equity_curve.py` (requires matplotlib ≥ 3.8)  
3. Compile LaTeX report using the template in `docs/report.tex`

---

## 7. Customization Guide

| Change Target        | How to Customize                         |
|---------------------|------------------------------------------|
| Ticker(s)           | Modify `tickers = [...]` in notebook cell 7 |
| Sentiment Model     | Change tokenizer in `BertTokenizer.from_pretrained(...)` |
| Technical Features  | Extend feature engineering in `fetch_data()` |
| Transaction Fee    | Modify `_take_action()` method in environment |
| Reward Function    | Adjust `step()` method to change reward calculation |
| Policy Network     | Configure SB3 `policy_kwargs` parameters |
| Hyperparameters    | Tune SB3 args or use `optuna-sb3` tuner |

---

## 8. Benchmarks & Limitations

- **Data leakage:** Single walk-forward, no out-of-sample testing  
- **Dummy headlines:** Sentiment scores are constant zero  
- **No slippage or fees:** Returns may be overstated (~50 bps per trade)  
- **Sparse reward:** Uses total net worth change, not per-step log returns  
- **Short training:** 10,000 PPO steps cover less than one epoch of data  

---


---

## 9. Conclusion

This project demonstrates the integration of financial news sentiment (via FinBERT) with a reinforcement learning trading agent (PPO) in a reproducible research pipeline. While the prototype achieves strong in-sample performance on AAPL, it is intended for educational and experimental purposes only. Users are encouraged to extend the framework by incorporating real news data, transaction costs, robust validation, and more diverse assets. Community contributions and feedback are welcome to help advance the project toward more realistic and generalizable trading research.






