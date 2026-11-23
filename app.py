import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from mmm import (CHANNELS, LABELS, HOLDOUT_WEEKS, channel_summary, fit, forecast,
                 load_data, marginal_roas, optimize_budget, weekly_response)

st.set_page_config(page_title="Media Mix Model & Budget Simulator", layout="wide")


@st.cache_resource
def get_model():
    df = load_data()
    return df, fit(df)


df, model = get_model()
current = {c: float(df[c].iloc[-52:].mean()) for c in CHANNELS}
total_now = sum(current.values())

st.title("Media Mix Model & Budget Simulator")
st.caption("Which ad channels actually drive revenue, and what happens if you move the budget around? "
           "Built on Meta Robyn's public simulated dataset: 208 weeks of spend across TV, out-of-home, "
           "print, Facebook, and search.")

m = model.metrics
c1, c2, c3, c4 = st.columns(4)
c1.metric("Holdout accuracy (MAPE)", f"{m['holdout_mape']:.1f}% error",
          help=f"Average error on the last {HOLDOUT_WEEKS} weeks, which the model never saw during training.")
c2.metric("Holdout R²", f"{m['holdout_r2']:.2f}")
c3.metric("Avg weekly media budget", f"${total_now:,.0f}", help="Average over the last 52 weeks.")
c4.metric("Channels modeled", len(CHANNELS))

tab1, tab2, tab3, tab4 = st.tabs(["What drives revenue", "Budget simulator", "Forecast", "How it works"])

# ---------- Tab 1 ----------
with tab1:
    summ = channel_summary(model, df)
    left, right = st.columns([1.1, 1])
    with left:
        st.subheader("Return on ad spend, last 52 weeks")
        fig = go.Figure(go.Bar(x=summ["roas"], y=summ["channel"], orientation="h",
                               text=[f"${v:.2f}" for v in summ["roas"]], textposition="outside"))
        fig.update_layout(height=320, margin=dict(l=0, r=40, t=10, b=0),
                          xaxis_title="Revenue per $1 spent", yaxis=dict(autorange="reversed"))
        st.plotly_chart(fig, width="stretch")
    with right:
        st.subheader("Where the next dollar works hardest")
        marg = pd.DataFrame({"Channel": [LABELS[c] for c in CHANNELS],
                             "Weekly spend now": [current[c] for c in CHANNELS],
                             "Revenue from next $1": [marginal_roas(model, c, current[c]) for c in CHANNELS],
                             "Ad effect lingers (weeks)": [1 / (1 - model.params[c]["decay"]) for c in CHANNELS]})
        marg = marg.sort_values("Revenue from next $1", ascending=False)
        st.dataframe(marg.style.format({"Weekly spend now": "${:,.0f}", "Revenue from next $1": "${:.2f}",
                                        "Ad effect lingers (weeks)": "{:.1f}"}),
                     hide_index=True, width="stretch")
        st.caption("Average return shows past performance. The next-dollar return shows where extra money "
                   "would help most, since every channel gets less efficient as spend goes up.")

    st.subheader("Weekly revenue: actual vs. model, split by source")
    contrib = model.contributions(df) * 1e6
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df["DATE"], y=contrib["baseline"], name="Baseline (no ads)",
                             stackgroup="one", line=dict(width=0)))
    for c in CHANNELS:
        fig.add_trace(go.Scatter(x=df["DATE"], y=contrib[c], name=LABELS[c], stackgroup="one", line=dict(width=0)))
    fig.add_trace(go.Scatter(x=df["DATE"], y=df["revenue"], name="Actual revenue", line=dict(color="black", width=2)))
    fig.add_vrect(x0=df["DATE"].iloc[-HOLDOUT_WEEKS], x1=df["DATE"].iloc[-1], fillcolor="gray", opacity=0.1,
                  annotation_text="holdout", annotation_position="top left")
    fig.update_layout(height=420, margin=dict(l=0, r=0, t=10, b=0), yaxis_title="Revenue ($)",
                      legend=dict(orientation="h", y=-0.15))
    st.plotly_chart(fig, width="stretch")

    st.subheader("Response curves: how revenue grows with weekly spend")
    fig = go.Figure()
    for c in CHANNELS:
        xs = np.linspace(0, current[c] * 3, 80)
        fig.add_trace(go.Scatter(x=xs, y=weekly_response(model, c, xs), name=LABELS[c]))
        fig.add_trace(go.Scatter(x=[current[c]], y=[weekly_response(model, c, current[c])], mode="markers",
                                 marker=dict(size=9), showlegend=False, hovertext=f"{LABELS[c]} today"))
    fig.update_layout(height=380, margin=dict(l=0, r=0, t=10, b=0), xaxis_title="Weekly spend ($)",
                      yaxis_title="Weekly revenue driven ($)")
    st.plotly_chart(fig, width="stretch")
    st.caption("Dots mark today's spend. Where a curve flattens, more money buys less extra revenue.")

# ---------- Tab 2 ----------
with tab2:
    st.subheader("Move the budget and see what happens")
    total = st.number_input("Total weekly media budget ($)", min_value=1000.0, value=float(round(total_now)),
                            step=1000.0)
    if "opt" not in st.session_state:
        st.session_state.opt = None
    if st.button("Find the best split for this budget"):
        st.session_state.opt = optimize_budget(model, current, total)
    start = st.session_state.opt or {c: current[c] / total_now * total for c in CHANNELS}

    cols = st.columns(len(CHANNELS))
    plan = {}
    for col, c in zip(cols, CHANNELS):
        plan[c] = col.slider(LABELS[c], 0.0, float(round(current[c] * 3, -2)), float(start[c]),
                             step=1.0, key=f"s_{c}_{round(start[c])}")
    plan_total = sum(plan.values())
    base_rev = sum(weekly_response(model, c, current[c]) for c in CHANNELS)
    new_rev = sum(weekly_response(model, c, plan[c]) for c in CHANNELS)

    a, b, c_ = st.columns(3)
    a.metric("Plan's weekly spend", f"${plan_total:,.0f}", f"{plan_total - total_now:+,.0f} vs today")
    b.metric("Weekly revenue from ads", f"${new_rev:,.0f}", f"{new_rev - base_rev:+,.0f} vs today")
    c_.metric("Overall return", f"${new_rev / plan_total:.2f} per $1" if plan_total else "n/a")

    comp = pd.DataFrame({"Channel": [LABELS[c] for c in CHANNELS],
                         "Today": [current[c] for c in CHANNELS], "Plan": [plan[c] for c in CHANNELS]})
    fig = go.Figure([go.Bar(name="Today", x=comp["Channel"], y=comp["Today"]),
                     go.Bar(name="Plan", x=comp["Channel"], y=comp["Plan"])])
    fig.update_layout(barmode="group", height=320, margin=dict(l=0, r=0, t=10, b=0), yaxis_title="Weekly spend ($)")
    st.plotly_chart(fig, width="stretch")
    st.caption("The optimizer keeps each channel between 30% and 300% of today's spend, so it won't "
               "suggest anything a real team couldn't actually do.")

    st.subheader("Plain-language summary")
    ranked = sorted(CHANNELS, key=lambda c: marginal_roas(model, c, plan[c]), reverse=True)
    moves = sorted(CHANNELS, key=lambda c: plan[c] - current[c])
    st.write(
        f"Under this plan, ads drive about **${new_rev:,.0f} a week**, "
        f"{'up' if new_rev >= base_rev else 'down'} **${abs(new_rev - base_rev):,.0f}** from today "
        f"({(new_rev / base_rev - 1) * 100:+.1f}%). "
        f"The biggest shift is {'into' if plan[moves[-1]] > current[moves[-1]] else 'out of'} "
        f"**{LABELS[moves[-1]]}** and away from **{LABELS[moves[0]]}**. "
        f"At these levels, the next dollar works hardest in **{LABELS[ranked[0]]}** "
        f"(about ${marginal_roas(model, ranked[0], plan[ranked[0]]):.2f} back) and least in "
        f"**{LABELS[ranked[-1]]}** (about ${marginal_roas(model, ranked[-1], plan[ranked[-1]]):.2f})."
    )

# ---------- Tab 3 ----------
with tab3:
    st.subheader("Next 12 weeks: today's mix vs. your plan")
    st.caption("Uses the plan from the Budget simulator tab. Assumes competitor sales and newsletters follow "
               "the same weeks last year, with no special events.")
    fc_today = forecast(model, df, current)
    fc_plan = forecast(model, df, plan)
    hist = df.iloc[-26:]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=hist["DATE"], y=hist["revenue"], name="Actual", line=dict(color="black")))
    fig.add_trace(go.Scatter(x=fc_today["DATE"], y=fc_today["forecast"], name="Forecast: today's mix",
                             line=dict(dash="dot")))
    fig.add_trace(go.Scatter(x=fc_plan["DATE"], y=fc_plan["forecast"], name="Forecast: your plan"))
    fig.update_layout(height=400, margin=dict(l=0, r=0, t=10, b=0), yaxis_title="Weekly revenue ($)",
                      legend=dict(orientation="h", y=-0.15))
    st.plotly_chart(fig, width="stretch")
    diff = fc_plan["forecast"].sum() - fc_today["forecast"].sum()
    st.metric("12-week revenue difference vs. today's mix", f"${diff:+,.0f}")

# ---------- Tab 4 ----------
with tab4:
    st.markdown(f"""
**The model.** Weekly revenue = baseline + the effect of each ad channel.

- **Baseline:** trend, yearly seasonality, competitor sales, newsletters, and special events.
- **Carryover (adstock):** an ad keeps working for a few weeks after it runs. Each channel gets its own decay rate.
- **Diminishing returns (Hill curve):** each extra dollar in a channel buys a little less than the last one.
- **Fitting:** a random search over carryover and saturation settings, with a linear fit for everything else.
  Ad effects are kept non-negative, so no channel can "hurt" sales just to make the math fit.

**Checking it.** The model was tuned on the first {len(df) - HOLDOUT_WEEKS} weeks and tested on the last
{HOLDOUT_WEEKS}, which it never saw. Holdout error: **{m['holdout_mape']:.1f}% MAPE**, R² **{m['holdout_r2']:.2f}**.

**Limits.** This is a simple, transparent MMM, not a full Bayesian one like Google's Meridian or Meta's Robyn.
It doesn't give uncertainty ranges, and the data is simulated, so treat the exact numbers as a demo.

**Data:** [Meta Robyn sample dataset](https://github.com/facebookexperimental/Robyn) (`dt_simulated_weekly`).
""")
