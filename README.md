# Media Mix Model & Budget Simulator

Which ad channels actually drive revenue, and what happens if you move the budget around?

This app fits a media mix model (MMM) on 208 weeks of ad spend and revenue, then lets you
reallocate the budget across TV, out-of-home, print, Facebook, and search and see the
predicted impact on revenue.

## What it does

- **What drives revenue:** return on ad spend by channel, the return on the *next* dollar,
  how long each channel's ads keep working, and weekly revenue split into baseline vs. each channel.
- **Budget simulator:** move spend between channels with sliders, or click one button to find the
  best split for a fixed budget. A plain-language summary explains the result.
- **Forecast:** projects the next 12 weeks under today's mix vs. your plan.

## How the model works

- **Carryover (adstock):** an ad keeps working for a few weeks after it runs.
- **Diminishing returns (Hill curve):** each extra dollar in a channel buys a little less.
- **Baseline:** trend, yearly seasonality, competitor sales, newsletters, and events.
- **Fit:** random search over carryover and saturation settings, with a constrained linear fit
  (ad effects can't go negative).
- **Validation:** tuned on the first 182 weeks, tested on the last 26 weeks it never saw.
  Holdout error is about 8% MAPE (R² 0.97).

## Data

Meta's open-source [Robyn](https://github.com/facebookexperimental/Robyn) sample dataset
(`dt_simulated_weekly`). It's simulated, so treat the exact numbers as a demo.

## Run it locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Limits

This is a simple, transparent MMM, not a full Bayesian model like Google's Meridian or Meta's
Robyn, so it doesn't give uncertainty ranges.
