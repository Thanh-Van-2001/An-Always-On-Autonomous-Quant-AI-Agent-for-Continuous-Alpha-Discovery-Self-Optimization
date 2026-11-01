# AQAA: An Always-On Autonomous Quant AI Agent

**Continuous Alpha Discovery, Self-Optimization, and Live Trading**

> Companion code for: *"AQAA: An Always-On Autonomous Quant AI Agent for Continuous Alpha Discovery, Self-Optimization, and Live Trading"* (preprint, 2026).

---

## Abstract

We introduce **AQAA**, a closed-loop autonomous system that continuously generates, validates, deploys, and retires quantitative alpha factors with minimal human intervention. AQAA integrates five cooperating modules -- a **Hypothesis Engine (HE)** that mines academic literature, news, and market data for investment theses; a **Code Synthesis and Backtest Agent (CSBA)** that translates hypotheses into executable factor code under a novel **Temporal Integrity Framework (TIF)** to eliminate LLM data-snooping bias; an **Adaptive Portfolio Optimizer (APO)** with regime-aware risk management; a **Live Execution and Monitoring Agent (LEMA)** with Bayesian IC tracking and an 8-state alpha lifecycle manager; and a **Feedback Loop Orchestrator (FLO)** that allocates compute across modules via a UCB1 multi-armed bandit.

In a 5-year backtest (2019-2024) across US equities, China A-shares, and cryptocurrency markets, AQAA achieves an annualized Sharpe ratio of **2.41** (US equities), **2.17** (China A-shares), and **2.73** (crypto), outperforming both traditional multi-factor baselines and recent LLM-based alpha-generation systems. The TIF protocol reduces suspected data-leakage rates from 23% to under 2%.

---

## System Architecture

```
+------------------------------------------------------------------+
|                     Feedback Loop Orchestrator (FLO)              |
|              UCB1 Bandit  |  Compute Budget Allocator             |
+----+----------+----------+----------+----------+-----------------+
     |          |          |          |          |
     v          v          v          v          v
+--------+ +---------+ +-------+ +--------+ +-----------+
|   HE   | |  CSBA   | |  APO  | |  LEMA  | |    KB     |
| Hypoth.| | Code    | | Regime| | Lifecy.| | Knowledge |
| Engine | | Synth + | | Aware | | Mgmt + | |   Base    |
|        | | TIF     | | Port. | | IC     | |           |
|        | | Backtest| | Optim.| | Watch  | |           |
+--------+ +---------+ +-------+ +--------+ +-----------+
     |          |          |          |
     v          v          v          v
 [Academic  [Factor   [Portfolio [Execution
  + News     Code +    Weights]   Orders +
  + Data]    Stats]              Monitoring]
```

**Figure 1 (simplified).** The five AQAA modules connected by the Feedback Loop Orchestrator. Arrows indicate data flow; the FLO continuously reallocates compute budget toward the module yielding the highest marginal IC gain per compute-hour.

---

## Modules

| Module | Package | Description |
|--------|---------|-------------|
| **HE** -- Hypothesis Engine | `aqaa.hypothesis_engine` | Mines academic papers, financial news, and market microstructure data for novel investment hypotheses. Uses LLM-based semantic embedding for novelty scoring (cosine-distance threshold tau_nov = 0.75). |
| **CSBA** -- Code Synthesis and Backtest Agent | `aqaa.csba` | Translates hypotheses into executable alpha-factor code via an iterative Writer/Judge LLM loop, then validates via a vectorized backtester. Enforces the **Temporal Integrity Framework (TIF)** -- a 4-step protocol comprising data partitioning, LLM-cutoff isolation, leakage detection tests, and live-validation gating. |
| **APO** -- Adaptive Portfolio Optimizer | `aqaa.apo` | Regime-aware portfolio construction. Contains a 3-state HMM regime detector (bull/bear/sideways), a mean-variance optimizer with crowding penalties, and fractional Kelly position sizing. |
| **LEMA** -- Live Execution and Monitoring Agent | `aqaa.lema` | Manages the full alpha lifecycle through an 8-state machine (hypothesis -> backtesting -> paper trading -> live -> review -> retired/mutated/archived). Includes Bayesian IC tracking via Kalman filtering, CUSUM + PELT change-point decay detection, factor crowding monitoring, and simulated TWAP execution. |
| **FLO** -- Feedback Loop Orchestrator | `aqaa.flo` | Outer control loop running a 24-hour autonomous cycle. Uses a UCB1 multi-armed bandit to dynamically allocate compute budget across HE, CSBA, APO, LEMA, and KB based on marginal IC gain per compute-hour. |

---

## Installation

### Prerequisites

- Python >= 3.9
- pip >= 23.0 (or conda)

### Core install (CPU only, no LLM or live trading)

```bash
git clone https://github.com/beetrade-tech/aqaa.git
cd aqaa
pip install -e .
```

### Full install (with LLM and live-trading support)

```bash
pip install -e ".[all]"
```

### Development install

```bash
pip install -e ".[all,dev]"
```

### Pinned dependencies (exact reproducibility)

```bash
pip install -r requirements.txt
pip install -e .
```

### Using Make

```bash
make install       # editable install with all optional + dev deps
make install-core  # core deps only
make install-dev   # core + dev deps
```

---

## Quick Start

### 1. Configuration

All hyperparameters are controlled via YAML configuration files. The defaults match the paper:

```bash
cat configs/default.yaml
```

Market-specific overrides are in `configs/markets/`:

```
configs/
  default.yaml              # Global defaults (Table 4 in the paper)
  markets/
    us_equities.yaml        # S&P 500 universe
    china_ashares.yaml      # CSI 300 universe
    crypto.yaml             # Top-50 crypto by market cap
```

### 2. Run a backtest

```bash
make run-backtest
# or directly:
python -m aqaa.experiments.backtest --config configs/default.yaml --output results/backtest/
```

### 3. Run the ablation study

```bash
make run-ablation
```

### 4. Run the TIF leakage-detection analysis

```bash
make run-tif-analysis
```

### 5. Run cross-market generalization

```bash
make run-cross-market
```

### 6. Run all experiments

```bash
make run-all
```

---

## Reproducing Paper Results

### Table 1 -- Main Results (Annualized, 2019-2024)

| Market | System | Sharpe | ARR (%) | MDD (%) | Calmar | IC | ICIR |
|--------|--------|--------|---------|---------|--------|------|------|
| US Equities | AQAA (ours) | **2.41** | **28.7** | 11.9 | **2.41** | **0.052** | **1.14** |
| | WorldQuant-101 | 1.03 | 12.1 | 18.4 | 0.66 | 0.031 | 0.62 |
| | AlphaGPT | 1.47 | 17.3 | 15.2 | 1.14 | 0.038 | 0.78 |
| | Alpha-GPT 2.0 | 1.62 | 19.1 | 14.1 | 1.35 | 0.041 | 0.84 |
| China A-shares | AQAA (ours) | **2.17** | **31.4** | 14.5 | **2.17** | **0.048** | **1.02** |
| | WorldQuant-101 | 0.89 | 10.5 | 22.1 | 0.48 | 0.027 | 0.54 |
| | AlphaGPT | 1.31 | 18.9 | 17.3 | 1.09 | 0.035 | 0.71 |
| Crypto | AQAA (ours) | **2.73** | **52.1** | 19.1 | **2.73** | **0.061** | **1.28** |
| | CCXT-Momentum | 0.74 | 15.3 | 31.2 | 0.49 | 0.022 | 0.41 |

### Table 2 -- Ablation Study (US Equities)

| Variant | Sharpe | Delta |
|---------|--------|-------|
| Full AQAA | 2.41 | -- |
| w/o TIF | 1.83 | -0.58 |
| w/o Regime-Aware APO | 2.04 | -0.37 |
| w/o IC Watchdog (LEMA) | 2.11 | -0.30 |
| w/o FLO (uniform budget) | 2.19 | -0.22 |
| w/o Novelty Filter (HE) | 2.08 | -0.33 |

### Table 3 -- TIF Effectiveness

| Metric | Without TIF | With TIF |
|--------|-------------|----------|
| Suspected leakage rate | 23.1% | 1.8% |
| Backtest-to-live Sharpe decay | 41.2% | 8.7% |
| False discovery rate (BH-corrected) | 18.4% | 3.2% |

---

## Running Tests

```bash
make test          # run all tests
make test-cov      # run tests with coverage
make lint          # ruff + mypy
make format        # auto-format with black + ruff --fix
make check         # lint + test
```

---

## Project Structure

```
aqaa/
  aqaa/                        # Main package
    __init__.py
    core/                      # Shared types, config loader
      types.py                 # Dataclasses: AlphaFactor, BacktestResult, etc.
      config.py                # YAML config loader
    hypothesis_engine/         # HE: hypothesis generation
    csba/                      # CSBA: code synthesis + backtest
      code_synthesizer.py      #   Writer/Judge LLM loop
      backtester.py            #   Vectorized backtesting engine
      tif.py                   #   Temporal Integrity Framework
      ldt.py                   #   Leakage Detection Test
    apo/                       # APO: portfolio optimization
      regime_detector.py       #   3-state HMM regime detector
      optimizer.py             #   Mean-variance optimizer
      kelly_sizing.py          #   Fractional Kelly sizing
    lema/                      # LEMA: execution + monitoring
      alm.py                   #   Alpha Lifecycle Manager (8-state FSM)
      bayesian_ic.py           #   Kalman-filter IC tracker
      decay_detector.py        #   CUSUM + PELT change-point detection
      crowding_detector.py     #   Factor crowding monitor
      executor.py              #   Order execution + risk checks
    flo/                       # FLO: feedback loop orchestrator
      ucb1.py                  #   UCB1 multi-armed bandit
      orchestrator.py          #   Main orchestration loop
  configs/
    default.yaml               # Default hyperparameters
    markets/                   # Per-market overrides
  data/                        # Data directory (not tracked in git)
  experiments/                 # Experiment entry points
  scripts/                     # Utility scripts
  tests/                       # Test suite
  results/                     # Output directory (not tracked in git)
```

---

## Citation

If you use AQAA in your research, please cite our paper:

```bibtex
@misc{aqaa2026,
  title  = {{AQAA}: An Always-On Autonomous Quant {AI} Agent for Continuous
            Alpha Discovery, Self-Optimization, and Live Trading},
  author = {{BeeTrade Technologies Inc}},
  year   = {2026},
  note   = {Preprint},
}
```

---

## License

This project is licensed under the **Apache License 2.0** -- see [LICENSE](LICENSE) for details.

Copyright 2024-2026 BeeTrade Technologies Inc.

---

## Disclaimer

This software is provided for **research purposes only**. It is not financial advice. Trading financial instruments carries risk, and past performance (including backtested results) does not guarantee future returns. Use at your own risk.
