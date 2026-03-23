# PCA-PPO-Portfolio-Optimization-with-Momentum-Filter-
This project trains a PPO reinforcement learning agent to optimize portfolio weights using PCA-derived market factors as state features.   The agent is trained to outperform an equal-weight benchmark under transaction costs and trading constraints.


## Motivation
Classical portfolio methods are often static. This project explores a dynamic allocation approach where:
- State = rolling PCA factors from asset returns
- Policy = PPO agent outputs portfolio weights
- Overlay = momentum-based filter to suppress weak assets
- Objective = beat equal-weight benchmark net of costs

## Method Overview
1. Load price data and convert to returns.
2. Split train/test data.
3. Fit PCA on train returns and project both train/test into factor space.
4. Train PPO agent in custom environment:
   - Rebalancing frequency
   - Transaction cost penalty
   - Minimum holding period
   - Minimum weight-change threshold
5. Backtest on unseen test data vs equal-weight portfolio.

## Project Structure
- `run.py` - end-to-end training and backtest entrypoint
- `data.py` - loading and return construction
- `factors.py` - PCA feature extraction
- `environment.py` - RL environment and reward logic
- `agent.py` - PPO policy/critic and training loop
- `backtest.py` - performance metrics and plots
- `sample_data.csv` - sample dataset

## Installation
```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS/Linux:
# source .venv/bin/activate

pip install -r requirements.txt
```

## Usage
```bash
python run.py
```

## Expected Output 
```bash
[INFO] Loading data from sample_data.csv
[INFO] Train period: 2015-01-01 to 2021-12-31 | Test period: 2022-01-01 to 2023-12-31
[INFO] PCA factors retained: 5 (explained variance: 82.4%)
[TRAIN] Episode 010 | Reward: 0.0124 | Turnover: 0.084 | Cost: 0.0011
[TRAIN] Episode 020 | Reward: 0.0189 | Turnover: 0.071 | Cost: 0.0009
...
[TEST] PPO CAGR: 14.2% | Sharpe: 1.18 | Max Drawdown: -11.6%
[TEST] EQW CAGR: 10.5% | Sharpe: 0.91 | Max Drawdown: -14.8%
```
<img src="C:\Users\Akhilesh Tayade\Downloads\Figure_1 (1).png" alt="Equity Curve" width="700"/>


