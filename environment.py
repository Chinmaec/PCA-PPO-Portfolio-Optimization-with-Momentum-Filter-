# RL PORTFOLIO ENVIRONMENT
import numpy as np
import pandas as pd

class Portfolio_Env: 

    """
    RL PORTFOLIO ENVIRONMENT

    CORE LOOP: state  → agent picks weights → portfolio return → reward → new state
    State  : last N days of PCA factors  
    Action : portfolio weights for each stock 
    Reward : portfolio return that day  (profit = positive, loss = negative)
    """

    def __init__(
        self,
        returns,
        pca_factors,
        lookback=20,
        transaction_cost=0.001,
        rebalance_every=1,          # keep this, but set to 1 in run.py
        min_holding_days=7,         # per-ticker minimum holding period
        min_weight_change=0.01      # 1% threshold (use 0.02 for 2%)
    ):
        self.returns = returns.values
        self.factors = pca_factors.values
        self.n_stocks = returns.shape[1]
        self.n_factors = pca_factors.shape[1]
        self.lookback = lookback
        self.T = len(returns)

        if (transaction_cost is None) or (transaction_cost < 0):
            transaction_cost = 0.001
        if (rebalance_every is None) or (rebalance_every < 1):
            rebalance_every = 1
        if (min_holding_days is None) or (min_holding_days < 1):
            min_holding_days = 7
        if (min_weight_change is None) or (min_weight_change < 0):
            min_weight_change = 0.01

        self.transaction_cost = float(transaction_cost)
        self.rebalance_every = int(rebalance_every)
        self.min_holding_days = int(min_holding_days)
        self.min_weight_change = float(min_weight_change)

        self.equal_weight = np.ones(self.n_stocks) / self.n_stocks
        self.reset()

    def _apply_trade_filters(self, proposed_weights):
        """
        Per-ticker filters:
        1) Only trade ticker i if min_holding_days has passed since last trade in i
        2) Only trade ticker i if abs(weight change) >= min_weight_change
        """
        days_since_trade = self.t - self.last_trade_t
        can_trade = days_since_trade >= self.min_holding_days
        big_enough = np.abs(proposed_weights - self.prev_weights) >= self.min_weight_change
        trade_mask = can_trade & big_enough

        if not np.any(trade_mask):
            return self.prev_weights.copy(), trade_mask

        new_weights = self.prev_weights.copy()

        # Keep non-traded tickers fixed; re-scale traded subset to use remaining budget.
        fixed_sum = new_weights[~trade_mask].sum()
        free_budget = 1.0 - fixed_sum
        trade_slice = proposed_weights[trade_mask]
        trade_sum = trade_slice.sum()

        if free_budget <= 0 or trade_sum <= 0:
            return self.prev_weights.copy(), np.zeros(self.n_stocks, dtype=bool)

        new_weights[trade_mask] = free_budget * (trade_slice / trade_sum)
        return new_weights, trade_mask


    # def _softmax(self, x):
    #     z = x - np.max(x)
    #     ez = np.exp(z)
    #     return ez / ez.sum()

    # different version of softmax 
    def _softmax(self, x, temperature=2.0):
        z = (x - np.max(x)) / temperature
        ez = np.exp(z)
        return ez / ez.sum()
        

    def should_rebalance_today(self):
        # First tradable day (t == lookback) is a rebalance day.
        day_idx = self.t - self.lookback
        return (day_idx % self.rebalance_every) == 0
    
    # State
    def get_state(self):
        "State : flattened PCA factors window over lookback period"
        "Shape : lookback x n_factors"

        window = self.factors[self.t - self.lookback : self.t] 
        return window.flatten()

    # Reset
    def reset(self):
        self.t = self.lookback
        self.portfolio_value = 1.0
        self.equal_value = 1.0

        self.prev_weights = np.ones(self.n_stocks) / self.n_stocks
        self.current_weights = self.prev_weights.copy()

        self.equal_prev_weights = self.equal_weight.copy()
        self.equal_current_weights = self.equal_prev_weights.copy()
        self.last_trade_t = np.full(self.n_stocks, self.t - self.min_holding_days, dtype= int)

        self.history = []
        return self.get_state()  
    
    def step(self,weights):
        """
        agent submits portfolio weights -> environment returns next state + reward
        weights : array of length n_stocks 
        returns : next state, reward, done)
        """
        traded = False
        turnover = 0.0
        tc = 0.0

        if self.should_rebalance_today() and weights is not None:
            # Agent raw proposal -> normalized
            target_weights = self._softmax(weights)

            # Momentum filter from lookback window
            returns_window = self.returns[self.t - self.lookback : self.t]
            momentum = returns_window.sum(axis=0)
            cut = np.percentile(momentum, 10)
            mask = (momentum > cut).astype(float)

            # Apply mask, then renormalize so capital stays fully invested
            target_weights = target_weights * mask
            wsum = target_weights.sum()
            if wsum > 0:
                target_weights = target_weights / wsum
            else:
                # Safety fallback if everything gets filtered out
                target_weights = self.prev_weights.copy()

            # 3) APPLY per-ticker holding + min-change filters (this was missing)
            proposed_weights = target_weights.copy()
            target_weights, trade_mask = self._apply_trade_filters(proposed_weights)

            turnover = np.abs(target_weights - self.prev_weights).sum()
            tc = self.transaction_cost * turnover
            traded = turnover > 0.0 # boolean: true

            if traded: 
                self.last_trade_t[trade_mask] = self.t
        else:
            target_weights = self.prev_weights.copy()


        # if self.should_rebalance_today() and weights is not None:
        #     target_weights = self._softmax(weights)
        #     turnover = np.abs(target_weights - self.prev_weights).sum()
        #     tc = self.transaction_cost * turnover
        #     traded = True
        # else:
        #     target_weights = self.prev_weights.copy()
   
        today_returns = self.returns[self.t]  

        # Agent portfolio return sum(w_i * r_i)
        port_return_gross = float(np.dot(target_weights, today_returns))
        port_return_net = port_return_gross - tc

        # equal weight benchmark (with same transaction cost logic)

        # equal_turnover = 0.0
        # equal_tc = 0.0
        # if self.should_rebalance_today():
        #     equal_target_weights = self.equal_weight.copy()
        #     equal_turnover = np.abs(equal_target_weights - self.equal_prev_weights).sum()
        #     equal_tc = self.transaction_cost * equal_turnover
        # else : 
        #     equal_target_weights = self.equal_prev_weights.copy()

        # equal_return_gross = float(np.dot(equal_target_weights, today_returns))
        # equal_return = equal_return_gross - equal_tc

        # Equal weight benchmark (rebalance daily)

        equal_target_weights = self.equal_weight.copy()

        equal_turnover = np.abs(equal_target_weights - self.equal_prev_weights).sum()
        equal_tc = self.transaction_cost * equal_turnover

        equal_return_gross = float(np.dot(equal_target_weights, today_returns))
        equal_return = equal_return_gross - equal_tc

        # Update portfolio values 
        self.portfolio_value *= (1+port_return_net)
        self.equal_value *= (1+equal_return)

        # weights drift after returns
        post = target_weights * (1.0 + today_returns)
        denom = post.sum()
        if denom > 0:
            self.current_weights = post / denom
        else:
            self.current_weights = target_weights.copy()

        # Benchmark drift after returns
        equal_post = equal_target_weights * (1.0 + today_returns)
        equal_denom = equal_post.sum()
        if equal_denom > 0:
            self.equal_current_weights = equal_post / equal_denom
        else:
            self.equal_current_weights = equal_target_weights.copy()

        # IMPORTANT carry-forward fix
        self.prev_weights = self.current_weights.copy()
        self.equal_prev_weights = self.equal_current_weights.copy()


        # Reward = agent's return − equal weight return
        # Positive reward = agent beat the benchmark today
        # Negative reward = agent underperformed benchmark today
        # reward = port_return_net - equal_return
        reward = (port_return_net - equal_return- 0.1 * turnover)

        # # Log 
        self.history.append({
            "t": self.t,
            "traded": traded,
            "turnover": turnover,
            "transaction_cost": tc,
            "weights": target_weights.copy(),
            "port_return_gross": port_return_gross,
            "port_return": port_return_net,   # net return after costs

            "equal_turnover": equal_turnover,
            "equal_transaction_cost": equal_tc,
            "equal_return_gross": equal_return_gross,
            "equal_return": equal_return,

            "port_value": self.portfolio_value,
            "equal_value": self.equal_value,
            "reward": reward
        })

        # Advance time 
        self.t += 1
        done = self.t >= self.T
        next_state = self.get_state() if not done else None

        return next_state, reward, done

    # Summary 
    def summary(self): 
        h = pd.DataFrame(self.history)
        pr = h["port_return"]
        er = h["equal_return"]

        def sharpe(r):
            return (r.mean() / r.std()) * np.sqrt(252) if r.std() > 0 else 0 
        
        print("=" * 45)
        print(f"{'':25} {'Agent':>8} {'EqWt': >8}")
        print(f"{'Final Value ($1 start)':<25} {h['port_value'].iloc[-1]:>8.3f} {h['equal_value'].iloc[-1]:>8.3f}")
        print(f"{'Ann. Return':<25} {pr.mean()*252:>8.1%} {er.mean()*252:>8.1%}")
        print(f"{'Ann. Volatility':<25} {pr.std()*np.sqrt(252):>8.1%} {er.std()*np.sqrt(252):>8.1%}")
        print(f"{'Sharpe Ratio':<25} {sharpe(pr):>8.2f} {sharpe(er):>8.2f}")
        print(f"{'Total Reward':<25} {h['reward'].sum():>8.3f}")
        print("=" * 45)

        return h