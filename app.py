import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from mmm import (CHANNELS, LABELS, HOLDOUT_WEEKS, channel_summary, fit, forecast,
                 load_data, marginal_roas, optimize_budget, weekly_response)

st.set_page_config(page_title="Ad Budget Simulator", page_icon="📊", layout="wide")

COLORS = {"tv_S": "#4C78A8", "ooh_S": "#F58518", "print_S": "#B279A2",
          "facebook_S": "#54A24B", "search_S": "#E45756"}
DIGITAL = ["facebook_S", "search_S"]
TRADITIONAL = ["tv_S", "ooh_S", "print_S"]


def money(x: float, sign: bool = False) -> str:
    s = "+" if sign and x > 0 else ("-" if x < 0 else "")
    x = abs(x)
    if x >= 1e6:
        v = f"${x / 1e6:.1f}M"
    elif x >= 1e3:
        v = f"${x / 1e3:.0f}K" if x >= 1e4 else f"${x / 1e3:.1f}K"
    else:
        v = f"${x:,.0f}"
    return s + v


@st.cache_resource
def get_model():
    df = load_data()
    return df, fit(df)


df, model = get_model()
today = {c: float(df[c].iloc[-52:].mean()) for c in CHANNELS}
today_total = sum(today.values())
best_today = optimize_budget(model, today, today_total)


def ads_revenue(plan: dict) -> float:
    return float(sum(weekly_response(model, c, plan[c]) for c in CHANNELS))


def set_plan(plan: dict):
    for c in CHANNELS:
        st.session_state[f"s_{c}"] = float(round(plan[c]))


if "s_tv_S" not in st.session_state:
    set_plan(today)

# ---------------- Sidebar: the controls ----------------
with st.sidebar:
    st.header("Your weekly ad budget")
    st.caption("Pick a quick scenario, or drag the sliders yourself. Everything on the page updates.")

    b1, b2 = st.columns(2)
    if b1.button("Today's mix", width="stretch"):
        set_plan(today)
    if b2.button("⭐ Best split", width="stretch", help="Same total budget, split the smartest way."):
        set_plan(best_today)
    b3, b4 = st.columns(2)
    if b3.button("Cut 20%", width="stretch", help="Same mix as today, 20% less money."):
        set_plan({c: today[c] * 0.8 for c in CHANNELS})
    if b4.button("More digital", width="stretch",
                 help="Move 20% of TV, billboard, and print money into Facebook and search."):
        moved = sum(today[c] * 0.2 for c in TRADITIONAL)
        dig_total = sum(today[c] for c in DIGITAL)
        p = {c: today[c] * 0.8 for c in TRADITIONAL}
        p.update({c: today[c] + moved * today[c] / dig_total for c in DIGITAL})
        set_plan(p)

    st.divider()
    plan = {}
    for c in CHANNELS:
        plan[c] = st.slider(f"{LABELS[c]}  (today: {money(today[c])})", 0.0, float(round(today[c] * 3, -2)),
                            step=100.0, key=f"s_{c}", format="$%d")
    plan_total = sum(plan.values())
    bd = plan_total - today_total
    st.metric("Total weekly budget", money(plan_total))
    st.caption(f"{money(bd, sign=True)} vs today" if abs(bd) > 50 else "Same as today")

    out = pd.DataFrame({"Channel": [LABELS[c] for c in CHANNELS],
                        "Weekly spend today": [round(today[c]) for c in CHANNELS],
                        "Weekly spend in plan": [round(plan[c]) for c in CHANNELS],
                        "Weekly revenue from ads (plan)": [round(float(weekly_response(model, c, plan[c])))
                                                           for c in CHANNELS]})
    st.download_button("Download this plan (CSV)", out.to_csv(index=False), "ad_budget_plan.csv",
                       width="stretch")

# ---------------- Header ----------------
st.title("📊 Ad Budget Simulator")
st.markdown("**Which ads actually bring in money, and what happens if you move the budget around?** "
            "This tool learned from 4 years of weekly ad spend and sales, so you can test budget ideas "
            "before spending a dollar.")

with st.expander("👋 New here? How to use this in 30 seconds", expanded=True):
    st.markdown(
        "1. **Look at the 3 cards below** for the main takeaways.\n"
        "2. **Use the sidebar on the left** (on a phone, tap » at the top left) to try a budget. "
        "Click **⭐ Best split** to see what the smartest split looks like.\n"
        "3. **Check the tabs** to see what changed, and what it means for the next 12 weeks.\n\n"
        "*The data is Meta's public sample dataset of a made-up company, so the numbers are a demo.*")

# ---------------- Headline cards ----------------
marg_today = {c: marginal_roas(model, c, today[c]) for c in CHANNELS}
best_ch = max(marg_today, key=marg_today.get)
worst_ch = min(marg_today, key=marg_today.get)
gain = ads_revenue(best_today) - ads_revenue(today)
def card(col, label, value, note):
    with col.container(border=True):
        st.caption(label)
        st.markdown(f"<div style='font-size:1.35rem;font-weight:600;line-height:1.25;margin:-4px 0 6px;white-space:nowrap'>{value}</div>",
                    unsafe_allow_html=True)
        st.caption(note)


k1, k2, k3 = st.columns(3)
card(k1, "🎯 Best next dollar", LABELS[best_ch],
     f"${marg_today[best_ch]:.2f} back per extra $1")
card(k2, "🧱 Most maxed out", LABELS[worst_ch],
     f"Only ${marg_today[worst_ch]:.2f} back per extra $1")
card(k3, "⭐ Smarter split", f"{money(gain, sign=True)}/week",
     f"{gain / ads_revenue(today) * 100:.0f}% more sales, same budget")

st.divider()
tab1, tab2, tab3, tab4 = st.tabs(["🧪 Your plan", "💡 What's working", "📈 Next 12 weeks", "❓ How it works"])

# ---------------- Tab 1: plan results ----------------
with tab1:
    rev_today, rev_plan = ads_revenue(today), ads_revenue(plan)
    if abs(rev_plan - rev_today) < 0.002 * rev_today:
        rev_plan = rev_today
    st.caption("Change the budget in the sidebar and these numbers update.")
    a, b, c_ = st.columns(3)
    a.metric("Weekly sales from ads", money(rev_plan),
             f"{money(rev_plan - rev_today, sign=True)} vs today" if rev_plan != rev_today else None)
    b.metric("Back per $1 spent", f"${rev_plan / plan_total:.2f}" if plan_total else "n/a",
             f"{(rev_plan / plan_total) - (rev_today / today_total):+.2f} vs today"
             if plan_total and rev_plan != rev_today else None)
    c_.metric("Change over a full year", money((rev_plan - rev_today) * 52, sign=True))

    # plain-language summary
    changes = sorted(CHANNELS, key=lambda c: plan[c] - today[c])
    up, down = changes[-1], changes[0]
    verdict = "more" if rev_plan > rev_today else ("less" if rev_plan < rev_today else "about the same")
    if verdict == "about the same":
        lines = ["**In plain words:** this plan brings in **about the same** sales from ads as today."]
    else:
        lines = [f"**In plain words:** this plan brings in **{verdict}** sales from ads than today "
                 f"({money(rev_plan - rev_today, sign=True)} a week)."]
    if plan[up] - today[up] > 50:
        lines.append(f"The biggest increase is **{LABELS[up]}** ({money(plan[up] - today[up], sign=True)} a week).")
    if today[down] - plan[down] > 50:
        lines.append(f"The biggest cut is **{LABELS[down]}** ({money(plan[down] - today[down], sign=True)} a week).")
    marg_plan = {c: marginal_roas(model, c, plan[c]) for c in CHANNELS}
    hi, lo = max(marg_plan, key=marg_plan.get), min(marg_plan, key=marg_plan.get)
    lines.append(f"If you had one more dollar, put it in **{LABELS[hi]}** (about ${marg_plan[hi]:.2f} back). "
                 f"**{LABELS[lo]}** gives the least for extra money (about ${marg_plan[lo]:.2f}).")
    st.info(" ".join(lines))

    st.markdown("**Weekly spend by channel: today vs. your plan**")
    fig = go.Figure()
    fig.add_trace(go.Bar(name="Today", x=[LABELS[c] for c in CHANNELS], y=[today[c] for c in CHANNELS],
                         marker_color="#C9C9C9"))
    fig.add_trace(go.Bar(name="Your plan", x=[LABELS[c] for c in CHANNELS], y=[plan[c] for c in CHANNELS],
                         marker_color=[COLORS[c] for c in CHANNELS]))
    fig.update_layout(barmode="group", height=340, margin=dict(l=0, r=0, t=10, b=0),
                      yaxis_title="Weekly spend ($)", legend=dict(orientation="h", y=-0.15, x=0))
    st.plotly_chart(fig, width="stretch")

# ---------------- Tab 2: what's working ----------------
with tab2:
    summ = channel_summary(model, df)
    st.markdown("**Two ways to judge a channel.** How much it earned on average last year, and how much "
                "*one more dollar* would earn today. They can be very different, because every channel "
                "pays off less as you spend more on it.")
    left, right = st.columns(2)
    with left:
        st.markdown("**Average: $ back per $1, last year**")
        fig = go.Figure(go.Bar(x=summ["roas"], y=summ["channel"], orientation="h",
                               marker_color=[COLORS[k] for k in summ["key"]],
                               text=[f"${v:.2f}" for v in summ["roas"]], textposition="outside"))
        fig.update_traces(cliponaxis=False)
        fig.update_layout(height=300, margin=dict(l=0, r=20, t=10, b=0),
                          yaxis=dict(autorange="reversed"),
                          xaxis=dict(showticklabels=False, showgrid=False, range=[0, summ["roas"].max() * 1.3]))
        st.plotly_chart(fig, width="stretch")
    with right:
        st.markdown("**Next dollar: $ back per extra $1 today**")
        order = sorted(CHANNELS, key=lambda c: marg_today[c], reverse=True)
        fig = go.Figure(go.Bar(x=[marg_today[c] for c in order], y=[LABELS[c] for c in order], orientation="h",
                               marker_color=[COLORS[c] for c in order],
                               text=[f"${marg_today[c]:.2f}" for c in order], textposition="outside"))
        fig.update_traces(cliponaxis=False)
        fig.update_layout(height=300, margin=dict(l=0, r=20, t=10, b=0),
                          yaxis=dict(autorange="reversed"),
                          xaxis=dict(showticklabels=False, showgrid=False, range=[0, max(marg_today.values()) * 1.3]))
        st.plotly_chart(fig, width="stretch")

    st.subheader("More spend, smaller gains")
    st.caption("Each line shows weekly sales from a channel as you spend more on it. The dot is where you are "
               "today. Where a line flattens out, extra money doesn't buy much.")
    fig = go.Figure()
    for c in CHANNELS:
        xs = np.linspace(0, today[c] * 3, 80)
        fig.add_trace(go.Scatter(x=xs, y=weekly_response(model, c, xs), name=LABELS[c],
                                 line=dict(color=COLORS[c], width=3)))
        fig.add_trace(go.Scatter(x=[today[c]], y=[weekly_response(model, c, today[c])], mode="markers",
                                 marker=dict(size=11, color=COLORS[c], line=dict(color="white", width=2)),
                                 showlegend=False, hoverinfo="skip"))
    fig.update_layout(height=380, margin=dict(l=0, r=0, t=10, b=0), xaxis_title="Weekly spend ($)",
                      yaxis_title="Weekly sales from this channel ($)", legend=dict(orientation="h", y=-0.2))
    st.plotly_chart(fig, width="stretch")

    st.subheader("Where weekly sales came from")
    st.caption("The gray area is sales the company would get with no ads at all. Each color on top is what "
               "a channel added. The black line is real sales.")
    contrib = model.contributions(df) * 1e6
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df["DATE"], y=contrib["baseline"], name="Without ads", stackgroup="one",
                             line=dict(width=0), fillcolor="rgba(180,180,180,0.6)"))
    for c in CHANNELS:
        fig.add_trace(go.Scatter(x=df["DATE"], y=contrib[c], name=LABELS[c], stackgroup="one",
                                 line=dict(width=0, color=COLORS[c])))
    fig.add_trace(go.Scatter(x=df["DATE"], y=df["revenue"], name="Real sales", line=dict(color="black", width=2)))
    fig.update_layout(height=400, margin=dict(l=0, r=0, t=10, b=0), yaxis_title="Weekly sales ($)",
                      legend=dict(orientation="h", y=-0.15))
    st.plotly_chart(fig, width="stretch")

# ---------------- Tab 3: forecast ----------------
with tab3:
    st.caption("What the next 12 weeks could look like with today's mix vs. your plan from the sidebar.")
    fc_today = forecast(model, df, today)
    fc_plan = forecast(model, df, plan)
    diff = fc_plan["forecast"].sum() - fc_today["forecast"].sum()
    if abs(diff) < 0.002 * fc_today["forecast"].sum():
        diff = 0.0
    st.metric("Extra sales over the next 12 weeks (your plan vs. today's mix)", money(diff, sign=True))
    hist = df.iloc[-26:]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=hist["DATE"], y=hist["revenue"], name="Past sales", line=dict(color="black")))
    x0, y0 = [hist["DATE"].iloc[-1]], [hist["revenue"].iloc[-1]]
    fig.add_trace(go.Scatter(x=x0 + list(fc_plan["DATE"]), y=y0 + list(fc_plan["forecast"]), name="Your plan", mode="lines",
                             line=dict(color="#E45756", width=3)))
    fig.add_trace(go.Scatter(x=x0 + list(fc_today["DATE"]), y=y0 + list(fc_today["forecast"]), name="Today's mix", mode="lines",
                             line=dict(dash="dot", color="#555", width=2)))
    fig.update_layout(height=400, margin=dict(l=0, r=0, t=10, b=0), yaxis_title="Weekly sales ($)",
                      legend=dict(orientation="h", y=-0.15))
    st.plotly_chart(fig, width="stretch")
    st.caption("Assumes competitors and email newsletters look like the same weeks last year, with no special events.")

# ---------------- Tab 4: how it works ----------------
with tab4:
    m = model.metrics
    st.markdown(f"""
**The idea.** Weekly sales = sales you'd get anyway + what each ad channel adds.

**Three things the model knows about ads:**
- **Ads keep working for a while.** A TV ad this week still nudges sales next week. Each channel fades at its own speed.
- **More spend, smaller gains.** The first $1,000 on a channel does more than the tenth $1,000.
- **Not everything is ads.** It also accounts for growth over time, seasons, competitors, email newsletters, and big events.

**Can you trust it?** I trained it on the first {len(df) - HOLDOUT_WEEKS} weeks, then tested it on the last
{HOLDOUT_WEEKS} weeks it had never seen. Its predictions were off by about **{m['holdout_mape']:.0f}%** on average.

**Limits.** This is a simple model built to be easy to understand. Bigger tools like Google's Meridian or
Meta's Robyn also show how confident each number is. The data is a made-up company, so treat the numbers as a demo.

**Data:** [Meta Robyn sample dataset](https://github.com/facebookexperimental/Robyn) · Built by Yin Yin Tan
""")
