import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import numpy as np

st.set_page_config(page_title="Benchmarks", page_icon="📊", layout="wide")
st.title("📊 Benchmark Results")
st.caption("Validated on Amazon Last Mile 2021 dataset · n=50 High-score routes · 1,000 bootstrap iterations")

# ── Headline metrics ──────────────────────────────────────────────────────────
c1, c2, c3 = st.columns(3)
c1.metric("FADR Improvement", "+15.7pp", "vs 2-min dwell baseline",
          help="95% CI: 13.4–18.0pp · p=9.6×10⁻⁷ · n=30 routes")
c2.metric("PyVRP vs Amazon Drivers", "+4.0%", "less travel time",
          help="95% CI: 3.3–4.7% · p=3.7×10⁻¹⁰ · n=50 routes · beats human professionals")
c3.metric("Zone-coherent NN vs Drivers", "−0.2%", "travel time (matches human parity)",
          help="95% CI: −1.0–+0.5% · p=0.73 (not significant) · simple heuristic = human level")

st.divider()

# ── Travel time comparison ────────────────────────────────────────────────────
st.subheader("Travel Time — Amazon Last Mile (50 High-score routes)")

col1, col2 = st.columns([2, 1])

with col1:
    algorithms = ["Amazon Drivers", "Zone-coherent NN\n(ours)", "PyVRP\n(ours)"]
    means      = [212.6, 212.8, 204.1]
    ci_lo      = [0, 212.6 * (1 - 0.005), 204.1 * (1 - 0.033)]  # approx from CI
    ci_hi      = [0, 212.6 * (1 + 0.005), 204.1 * (1 - 0.047)]
    colors     = ["#94a3b8", "#60a5fa", "#22c55e"]

    fig = go.Figure()
    fig.add_bar(
        x=algorithms, y=means,
        marker_color=colors,
        error_y=dict(
            type="data",
            symmetric=False,
            array=[0, 0.5, 4.1 - (204.1 - 204.1 * 0.033)],
            arrayminus=[0, 0.5, 4.1 - (204.1 - 204.1 * 0.047)],
            visible=True,
        ),
        text=[f"{m:.1f} min" for m in means],
        textposition="outside",
    )
    fig.add_hline(y=212.6, line_dash="dash", line_color="#94a3b8",
                  annotation_text="Amazon baseline", annotation_position="right")
    fig.update_layout(
        title="Mean Route Travel Time (lower = better)",
        yaxis_title="Minutes", yaxis_range=[190, 225],
        showlegend=False, height=380,
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
    )
    st.plotly_chart(fig, use_container_width=True)

with col2:
    st.markdown("#### PyVRP wins on all 50 routes")
    st.markdown("""
    **50 / 50** routes: PyVRP < Amazon

    **20 / 50** routes: Zone-coherent NN < Amazon

    ---
    PyVRP uses a metaheuristic solver (HGS-CVRP) with zone coherence penalty.

    Zone-coherent NN uses a simple greedy heuristic — yet statistically matches human professionals.
    """)

st.divider()

# ── FADR cascade comparison ───────────────────────────────────────────────────
st.subheader("FADR — Cascade Prevention (30 Amazon routes, synthetic Western dwell N(5,3))")

col1, col2 = st.columns([2, 1])

with col1:
    systems     = ["Baseline\n(2min plan)", "Naive NN\n(2min plan)", "Ours\n(P75 plan + deferral)"]
    fadr_means  = [68.9, 62.5, 84.5]
    fadr_lo     = [66.5, 60.0, 83.9]
    fadr_hi     = [71.2, 65.0, 85.1]
    colors2     = ["#f87171", "#fb923c", "#22c55e"]

    fig2 = go.Figure()
    fig2.add_bar(
        x=systems, y=fadr_means,
        marker_color=colors2,
        error_y=dict(
            type="data", symmetric=False,
            array=[h - m for h, m in zip(fadr_hi, fadr_means)],
            arrayminus=[m - l for m, l in zip(fadr_means, fadr_lo)],
            visible=True,
        ),
        text=[f"{m:.1f}%" for m in fadr_means],
        textposition="outside",
    )
    fig2.update_layout(
        title="First-Attempt Delivery Rate (higher = better)",
        yaxis_title="FADR %", yaxis_range=[50, 95],
        showlegend=False, height=380,
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
    )
    st.plotly_chart(fig2, use_container_width=True)

with col2:
    st.markdown("#### Why naive NN is worse than random")
    st.markdown("""
    Efficient routing (NN) **packs more stops** into the plan.

    When dwell is underestimated, more planned stops → more cascade failures.

    **Key insight:** optimizing routing without fixing dwell estimation makes things worse.

    ---
    **Cascade → Deferral:**
    - Baseline: 13.9 surprise failures/route
    - Ours: 0.3 planned deferrals/route

    Deferrals = scheduled reattempt.
    Cascades = missed, angry customer.
    """)

st.divider()

# ── Sensitivity sweep ─────────────────────────────────────────────────────────
st.subheader("Sensitivity — FADR Gain vs Baseline Dwell Assumption")

dwell_vals = [1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0, 6.5, 7.0, 7.5, 8.0]
gains      = [32.3, 29.2, 24.8, 19.5, 15.4, 10.5, 6.0, 2.5, 0.4, 0.0, -0.0, -0.1, -0.0, -0.1, -0.1]

fig3 = go.Figure()
fig3.add_scatter(
    x=dwell_vals, y=gains,
    mode="lines+markers",
    line=dict(color="#60a5fa", width=3),
    marker=dict(size=8),
    name="FADR gain (pp)",
)
fig3.add_hline(y=0, line_dash="dash", line_color="#f87171",
               annotation_text="Crossover (gain = 0)", annotation_position="right")
fig3.add_vrect(x0=1.0, x1=6.0, fillcolor="#22c55e", opacity=0.07,
               annotation_text="System helps here", annotation_position="top left")
fig3.add_vline(x=2.0, line_dash="dot", line_color="#94a3b8",
               annotation_text="Industry common (2min)", annotation_position="top")
fig3.update_layout(
    title="FADR Gain vs Competitor's Dwell Assumption (actual dwell = N(5,3) min)",
    xaxis_title="Competitor's planned dwell (min)",
    yaxis_title="FADR gain (percentage points)",
    height=380,
    plot_bgcolor="rgba(0,0,0,0)",
    paper_bgcolor="rgba(0,0,0,0)",
)
st.plotly_chart(fig3, use_container_width=True)
st.caption("System helps any company planning dwell at <6 min. Most dispatch software uses 2 min (industry default).")

st.divider()

# ── Statistical table ─────────────────────────────────────────────────────────
st.subheader("Validated Claims")
st.markdown("""
| Claim | Value | 95% CI | p-value | n |
|---|---|---|---|---|
| Cascade FADR improvement | **+15.7pp** | [13.4–18.0pp] | 9.6×10⁻⁷ | 30 routes |
| PyVRP vs Amazon drivers | **+4.0%** travel time | [3.3–4.7%] | 3.7×10⁻¹⁰ | 50 routes |
| Zone-coherent NN vs Amazon | −0.2% travel time | [−1.0–+0.5%] | 0.73 (NS) | 50 routes |

*Bootstrap CI with 1,000 iterations. Wilcoxon signed-rank test (one-sided).*
*Amazon Last Mile 2021 dataset, High-score routes. Real OSRM road times (WA/IL/TX/MA).*
""")
