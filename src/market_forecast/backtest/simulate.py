"""A friction check on the signal, not a trading strategy.

The question is narrow: does a directional edge of this size survive realistic costs?
The answer is reported as a break-even cost in basis points, which is comparable across
signals and cannot be inflated by leverage or position sizing the way a return figure can.

Everything here understates real-world difficulty. Fills are assumed at the close,
capacity and market impact are ignored, borrow for the short leg is free, and the
universe contains only companies that still exist.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

TRADING_DAYS = 252
BPS = 1e-4


@dataclass
class BacktestResult:
    equity: pd.Series
    daily_returns: pd.Series
    cost_bps: float
    n_sessions: int
    mean_positions: float
    turnover: float
    total_return: float
    annualised_return: float
    annualised_volatility: float
    sharpe: float
    max_drawdown: float
    hit_rate: float
    metadata: dict[str, float] = field(default_factory=dict)

    def summary(self) -> str:
        return (
            f"cost {self.cost_bps:.0f}bps: ann return {self.annualised_return:+.2%}, "
            f"vol {self.annualised_volatility:.2%}, sharpe {self.sharpe:+.2f}, "
            f"max dd {self.max_drawdown:.2%}, turnover {self.turnover:.2f}/day"
        )


def _metrics(
    daily: pd.Series, cost_bps: float, positions: pd.Series, turnover: float
) -> BacktestResult:
    equity = (1.0 + daily).cumprod()
    drawdown = equity / equity.cummax() - 1.0
    volatility = float(daily.std(ddof=1) * np.sqrt(TRADING_DAYS)) if len(daily) > 1 else 0.0
    mean = float(daily.mean() * TRADING_DAYS)
    return BacktestResult(
        equity=equity,
        daily_returns=daily,
        cost_bps=cost_bps,
        n_sessions=int(len(daily)),
        mean_positions=float(positions.mean()),
        turnover=turnover,
        total_return=float(equity.iloc[-1] - 1.0) if len(equity) else 0.0,
        annualised_return=mean,
        annualised_volatility=volatility,
        sharpe=float(mean / volatility) if volatility > 0 else 0.0,
        max_drawdown=float(drawdown.min()) if len(drawdown) else 0.0,
        hit_rate=float((daily > 0).mean()) if len(daily) else 0.0,
    )


def simulate(
    predictions: pd.DataFrame,
    returns: pd.Series,
    threshold: float = 0.5,
    cost_bps: float = 0.0,
) -> BacktestResult:
    """Equal weight across every name whose probability clears the threshold.

    Positions are unlevered and long only: a name is either held at equal weight or not
    held. Costs are charged on the change in weight, so a name held for several sessions
    pays once on entry and once on exit.
    """
    frame = predictions[["prob"]].copy()
    frame["ret"] = returns.reindex(frame.index)
    frame = frame.dropna()
    if frame.empty:
        raise ValueError("no overlapping predictions and returns to simulate")

    frame["signal"] = (frame["prob"] >= threshold).astype("float64")
    weights = frame["signal"].unstack("ticker").fillna(0.0)
    active = weights.sum(axis=1).replace(0.0, np.nan)
    weights = weights.div(active, axis=0).fillna(0.0)

    asset_returns = frame["ret"].unstack("ticker").reindex_like(weights).fillna(0.0)
    gross = (weights * asset_returns).sum(axis=1)

    traded = weights.diff().abs().sum(axis=1)
    traded.iloc[0] = weights.iloc[0].abs().sum()
    net = gross - traded * cost_bps * BPS

    positions = (weights > 0).sum(axis=1)
    result = _metrics(net, cost_bps, positions, float(traded.mean()))
    result.metadata["gross_annualised_return"] = float(gross.mean() * TRADING_DAYS)
    result.metadata["threshold"] = threshold
    return result


def cost_sweep(
    predictions: pd.DataFrame,
    returns: pd.Series,
    threshold: float = 0.5,
    costs: tuple[float, ...] = (0.0, 5.0, 10.0, 20.0),
) -> pd.DataFrame:
    rows = []
    for cost in costs:
        result = simulate(predictions, returns, threshold, cost)
        rows.append(
            {
                "cost_bps": cost,
                "annualised_return": result.annualised_return,
                "annualised_vol": result.annualised_volatility,
                "sharpe": result.sharpe,
                "max_drawdown": result.max_drawdown,
                "turnover_per_day": result.turnover,
                "mean_positions": result.mean_positions,
            }
        )
    return pd.DataFrame(rows).set_index("cost_bps")


def break_even_cost_bps(
    predictions: pd.DataFrame, returns: pd.Series, threshold: float = 0.5
) -> float:
    """Round-trip cost in basis points at which the strategy stops making money.

    The single most useful number in this section. A break-even below realistic retail
    costs means the edge is not tradable however good the classification metrics look.
    """
    gross = simulate(predictions, returns, threshold, cost_bps=0.0)
    if gross.turnover <= 0:
        return 0.0
    daily_edge = gross.daily_returns.mean()
    if daily_edge <= 0:
        return 0.0
    return float(daily_edge / (gross.turnover * BPS))


def buy_and_hold(returns: pd.Series) -> BacktestResult:
    """Equal-weighted, always invested. The comparison any strategy has to earn against."""
    wide = returns.unstack("ticker")
    daily = wide.mean(axis=1).dropna()
    positions = wide.notna().sum(axis=1)
    return _metrics(daily, 0.0, positions, 0.0)
