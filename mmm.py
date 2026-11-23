"""Media Mix Model with carryover (adstock) and diminishing returns (Hill saturation).

Data: Meta Robyn's public simulated weekly dataset (208 weeks of spend by channel and revenue).
Model: revenue = baseline (trend + seasonality + controls) + sum over channels of
       beta_c * hill(adstock(spend_c)), with every beta_c >= 0.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy.optimize import lsq_linear, minimize

CHANNELS = ["tv_S", "ooh_S", "print_S", "facebook_S", "search_S"]
LABELS = {"tv_S": "TV", "ooh_S": "Out-of-home", "print_S": "Print",
          "facebook_S": "Facebook", "search_S": "Search"}
HOLDOUT_WEEKS = 26
SEED = 7


# ---------- transforms ----------
def adstock(x: np.ndarray, decay: float) -> np.ndarray:
    """Geometric carryover: part of each week's ad effect spills into later weeks."""
    out = np.zeros_like(x, dtype=float)
    carry = 0.0
    for i, v in enumerate(x):
        carry = v + decay * carry
        out[i] = carry
    return out


def hill(x: np.ndarray, half_sat: float, shape: float) -> np.ndarray:
    """Diminishing returns: response rises fast, then flattens. Output is 0 to 1."""
    x = np.maximum(x, 0)
    return x**shape / (x**shape + half_sat**shape)


# ---------- design matrix ----------
def load_data(path: str = "data_weekly.csv") -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["DATE"])
    df["event_flag"] = (df["events"].astype(str) != "na").astype(float)
    return df


def baseline_features(df: pd.DataFrame, t0: int = 0) -> pd.DataFrame:
    n = len(df)
    t = np.arange(t0, t0 + n)
    feats = {"intercept": np.ones(n), "trend": t / 52.0}
    for k in (1, 2):
        feats[f"sin{k}"] = np.sin(2 * np.pi * k * t / 52.0)
        feats[f"cos{k}"] = np.cos(2 * np.pi * k * t / 52.0)
    feats["competitor_sales"] = df["competitor_sales_B"].to_numpy() / 1e6
    feats["newsletter"] = df["newsletter"].to_numpy() / 1e4
    feats["event_flag"] = df["event_flag"].to_numpy()
    return pd.DataFrame(feats)


@dataclass
class MMM:
    params: dict = field(default_factory=dict)      # channel -> {decay, half_sat, shape}
    coefs: dict = field(default_factory=dict)       # feature -> coef
    metrics: dict = field(default_factory=dict)
    channel_max: dict = field(default_factory=dict)  # scale for half_sat

    # media block for given params
    def media_block(self, df: pd.DataFrame, params: dict | None = None) -> pd.DataFrame:
        params = params or self.params
        cols = {}
        for c in CHANNELS:
            p = params[c]
            ad = adstock(df[c].to_numpy(float), p["decay"])
            cols[c] = hill(ad, p["half_sat"], p["shape"])
        return pd.DataFrame(cols)

    def design(self, df, params=None, t0=0):
        return pd.concat([baseline_features(df, t0).reset_index(drop=True),
                          self.media_block(df, params).reset_index(drop=True)], axis=1)

    def predict(self, df, t0=0) -> np.ndarray:
        X = self.design(df, t0=t0)
        return X.to_numpy() @ np.array([self.coefs[c] for c in X.columns])

    def contributions(self, df, t0=0) -> pd.DataFrame:
        X = self.design(df, t0=t0)
        out = X * pd.Series(self.coefs)[X.columns]
        base_cols = [c for c in X.columns if c not in CHANNELS]
        res = out[CHANNELS].copy()
        res["baseline"] = out[base_cols].sum(axis=1)
        return res


def _fit_linear(X: pd.DataFrame, y: np.ndarray):
    lb = np.array([0.0 if c in CHANNELS else -np.inf for c in X.columns])
    ub = np.full(len(X.columns), np.inf)
    r = lsq_linear(X.to_numpy(), y, bounds=(lb, ub), method="trf")
    return dict(zip(X.columns, r.x))


def _score(y, yhat):
    resid = y - yhat
    r2 = 1 - np.sum(resid**2) / np.sum((y - y.mean()) ** 2)
    mape = np.mean(np.abs(resid) / y) * 100
    return r2, mape


def fit(df: pd.DataFrame, n_trials: int = 2500) -> MMM:
    """Random search over carryover/saturation settings, linear fit for the rest.
    Picks the setting with the best fit on training weeks, then reports
    accuracy on the last 26 weeks, which the model never saw."""
    rng = np.random.default_rng(SEED)
    train = df.iloc[:-HOLDOUT_WEEKS].reset_index(drop=True)
    test = df.iloc[-HOLDOUT_WEEKS:].reset_index(drop=True)
    y_tr = train["revenue"].to_numpy() / 1e6
    model = MMM()
    model.channel_max = {c: float(df[c].mean()) for c in CHANNELS}

    best = None
    for _ in range(n_trials):
        params = {}
        for c in CHANNELS:
            decay = rng.uniform(0.0, 0.8) if c in ("tv_S", "ooh_S", "print_S") else rng.uniform(0.0, 0.4)
            params[c] = {"decay": decay,
                         "half_sat": model.channel_max[c] / (1 - decay) * rng.uniform(0.3, 2.5),
                         "shape": rng.choice([1.0, 1.5, 2.0])}
        X = model.design(train, params)
        coefs = _fit_linear(X, y_tr)
        yhat = X.to_numpy() @ np.array(list(coefs.values()))
        sse = np.sum((y_tr - yhat) ** 2)
        if best is None or sse < best[0]:
            best = (sse, params, coefs)

    model.params, model.coefs = best[1], best[2]
    yhat_tr = model.predict(train)
    yhat_te = model.predict(test, t0=len(train))
    r2_tr, mape_tr = _score(y_tr, yhat_tr)
    r2_te, mape_te = _score(test["revenue"].to_numpy() / 1e6, yhat_te)
    model.metrics = {"train_r2": r2_tr, "train_mape": mape_tr,
                     "holdout_r2": r2_te, "holdout_mape": mape_te,
                     "holdout_weeks": HOLDOUT_WEEKS}

    # refit linear part on all weeks with the chosen carryover/saturation settings
    X_all = model.design(df)
    model.coefs = _fit_linear(X_all, df["revenue"].to_numpy() / 1e6)
    return model


# ---------- insights ----------
def channel_summary(model: MMM, df: pd.DataFrame, last_n: int = 52) -> pd.DataFrame:
    contrib = model.contributions(df).iloc[-last_n:]
    spend = df[CHANNELS].iloc[-last_n:].sum()
    rows = []
    for c in CHANNELS:
        rev = contrib[c].sum() * 1e6
        rows.append({"channel": LABELS[c], "key": c, "spend": spend[c],
                     "revenue_driven": rev,
                     "roas": rev / spend[c] if spend[c] > 0 else np.nan,
                     "decay": model.params[c]["decay"]})
    return pd.DataFrame(rows).sort_values("roas", ascending=False)


def weekly_response(model: MMM, c: str, weekly_spend: np.ndarray | float) -> np.ndarray:
    """Revenue per week once a steady weekly spend has been running a while."""
    p = model.params[c]
    steady = np.asarray(weekly_spend, float) / (1 - p["decay"])
    return model.coefs[c] * 1e6 * hill(steady, p["half_sat"], p["shape"])


def marginal_roas(model: MMM, c: str, weekly_spend: float, step: float = 100.0) -> float:
    return float((weekly_response(model, c, weekly_spend + step) - weekly_response(model, c, weekly_spend)) / step)


def optimize_budget(model: MMM, current: dict, total: float, lo: float = 0.3, hi: float = 3.0) -> dict:
    """Split a fixed weekly budget across channels to get the most revenue."""
    keys = CHANNELS
    x0 = np.array([current[c] for c in keys], float)
    x0 = x0 / x0.sum() * total
    bounds = [(current[c] * lo, max(current[c] * hi, 1.0)) for c in keys]

    def neg(x):
        return -sum(weekly_response(model, c, x[i]) for i, c in enumerate(keys))

    r = minimize(neg, x0, bounds=bounds, method="SLSQP",
                 constraints=[{"type": "eq", "fun": lambda x: x.sum() - total}])
    return dict(zip(keys, r.x))


def forecast(model: MMM, df: pd.DataFrame, weekly_plan: dict, weeks: int = 12) -> pd.DataFrame:
    """Project the next N weeks under a spend plan. Assumptions: competitor sales and
    newsletter follow the same weeks last year, no special events."""
    last = df.iloc[-52:].reset_index(drop=True)
    fut = last.iloc[:weeks].copy().reset_index(drop=True)
    fut["DATE"] = [df["DATE"].iloc[-1] + pd.Timedelta(weeks=i + 1) for i in range(weeks)]
    fut["event_flag"] = 0.0
    for c in CHANNELS:
        fut[c] = weekly_plan[c]
    hist = pd.concat([df, fut], ignore_index=True)
    pred = model.predict(hist)[-weeks:] * 1e6
    return pd.DataFrame({"DATE": fut["DATE"], "forecast": pred})


if __name__ == "__main__":
    df = load_data()
    m = fit(df)
    print(json.dumps(m.metrics, indent=2))
    print(json.dumps(m.params, indent=2, default=float))
    print(channel_summary(m, df).to_string(index=False))
    cur = {c: float(df[c].iloc[-52:].mean()) for c in CHANNELS}
    tot = sum(cur.values())
    opt = optimize_budget(m, cur, tot)
    gain = sum(weekly_response(m, c, opt[c]) for c in CHANNELS) - sum(weekly_response(m, c, cur[c]) for c in CHANNELS)
    print("current", {LABELS[c]: round(v) for c, v in cur.items()})
    print("optimal", {LABELS[c]: round(v) for c, v in opt.items()})
    print("weekly media revenue gain", round(gain))
    print(forecast(m, df, cur).head())
