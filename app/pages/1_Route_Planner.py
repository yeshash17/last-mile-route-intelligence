"""
Page 1 — Route Planner
Upload stops CSV → compare baseline vs optimised plan → FADR + Monte Carlo risk.
"""

import sys
import io
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import folium
from streamlit_folium import st_folium

from models.monte_carlo import simulate_route_risk
from data.distance_matrix import build_matrix

st.set_page_config(page_title="Route Planner", page_icon="🗺️", layout="wide")
st.title("🗺️ Route Planner")
st.caption("Upload your delivery stops — see how smarter planning prevents cascade failures and saves money")

PLANNED_DWELL_BASELINE = 2.0

SCENARIOS_DIR = Path(__file__).parent.parent.parent / "scenarios"

SCENARIO_FILES = {
    "🏙️ Dense Urban (high cascade risk)":     "scenario_1_dense_urban.csv",
    "🏡 Suburban Spread (moderate risk)":      "scenario_2_suburban_spread.csv",
    "🏢 Business District (low risk)":         "scenario_3_business_district.csv",
    "😱 Nightmare Route (dramatic contrast)":  "scenario_4_nightmare_route.csv",
    "📋 Default 40-stop Seattle sample":       None,
}

SAMPLE_CSV = """lat,lng,label,dwell_mins,p_success
47.6062,-122.3321,Capitol Hill 1,5.0,0.90
47.6092,-122.3401,Capitol Hill 2,4.0,0.85
47.6001,-122.3291,Beacon Hill 1,6.5,0.80
47.6150,-122.3450,Queen Anne 1,5.0,0.88
47.6080,-122.3350,Downtown Office,3.5,0.93
47.5990,-122.3380,SoDo Warehouse,6.0,0.82
47.6120,-122.3280,First Hill 1,5.5,0.87
47.6030,-122.3420,Pioneer Square,4.5,0.85
47.6200,-122.3310,Lower Queen Anne,6.0,0.79
47.6070,-122.3470,Belltown 1,5.0,0.88
47.6140,-122.3390,Madison Valley,4.5,0.91
47.5970,-122.3360,SODO Business,6.5,0.78
47.6110,-122.3440,Belltown 2,4.0,0.86
47.6050,-122.3300,Intl District,5.5,0.83
47.6180,-122.3480,South Lake Union,4.0,0.93
47.6020,-122.3250,Georgetown,5.0,0.84
47.6090,-122.3340,Central District,6.0,0.81
47.6160,-122.3270,Eastlake 1,4.5,0.89
47.6035,-122.3460,Pioneer Square 2,5.5,0.82
47.6125,-122.3315,Capitol Hill 3,4.0,0.87
47.5980,-122.3410,SODO Studio,5.0,0.85
47.6070,-122.3230,Madrona,6.0,0.80
47.6155,-122.3425,Westlake,4.5,0.91
47.6010,-122.3340,Columbia City,5.0,0.84
47.6095,-122.3500,Fremont,6.5,0.77
47.6045,-122.3355,First Hill 2,5.0,0.86
47.6175,-122.3305,Eastlake 2,4.5,0.90
47.6085,-122.3415,Belltown 3,5.5,0.83
47.5965,-122.3325,Georgetown 2,6.0,0.79
47.6130,-122.3475,Lower Queen Anne 2,4.0,0.88
47.6015,-122.3280,Beacon Hill 2,5.5,0.82
47.6100,-122.3360,Capitol Hill 4,5.0,0.87
47.6055,-122.3440,Pioneer Square 3,4.5,0.84
47.6185,-122.3395,South Lake Union 2,4.0,0.92
47.6075,-122.3260,Madrona 2,6.0,0.80
47.6145,-122.3340,Madison Valley 2,5.0,0.89
47.5985,-122.3460,SODO 2,6.5,0.76
47.6025,-122.3395,Georgetown 3,5.5,0.83
47.6165,-122.3455,Queen Anne 2,4.5,0.88
47.6105,-122.3295,First Hill 3,5.0,0.86
"""

# ── Helpers ───────────────────────────────────────────────────────────────────

def haversine_km(lat1, lng1, lat2, lng2):
    R = 6371.0
    la1, lo1 = np.radians(lat1), np.radians(lng1)
    la2, lo2 = np.radians(lat2), np.radians(lng2)
    dlat, dlng = la2 - la1, lo2 - lo1
    a = np.sin(dlat/2)**2 + np.cos(la1)*np.cos(la2)*np.sin(dlng/2)**2
    return R * 2 * np.arcsin(np.sqrt(a))


def build_haversine_matrix(coords):
    n = len(coords)
    mat = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if i != j:
                km = haversine_km(coords[i][0], coords[i][1], coords[j][0], coords[j][1])
                mat[i, j] = km / 30.0 * 60
    return mat


def nn_order(n, mat):
    unvisited = set(range(1, n+1))
    order, cur = [], 0
    while unvisited:
        nxt = min(unvisited, key=lambda j: mat[cur, j])
        order.append(nxt)
        unvisited.remove(nxt)
        cur = nxt
    return order


def plan_trim(full_order, mat, planned_dwell, shift_mins):
    planned, t = [], 0.0
    for i in full_order:
        travel = float(mat[0 if not planned else planned[-1], i])
        t += travel + planned_dwell
        if t + mat[i, 0] > shift_mins:
            break
        planned.append(i)
    return planned


def simulate(order, mat, dwells, p_success, shift_mins):
    t, completed, cascade = 0.0, [], []
    for pos, i in enumerate(order):
        prev = 0 if pos == 0 else order[pos-1]
        t_arr = t + mat[prev, i]
        if t_arr >= shift_mins:
            cascade.extend(order[pos:])
            break
        t = t_arr + dwells[i-1]
        completed.append(i)
    delivered = sum(p_success[i-1] for i in completed)
    total     = len(completed) + len(cascade)
    fadr      = delivered / total if total > 0 else 0.0
    return {"fadr": fadr, "completed": len(completed),
            "cascade": len(cascade), "delivered": delivered}


# ── Sidebar ───────────────────────────────────────────────────────────────────

with st.sidebar:
    st.header("⚙️ Settings")

    st.subheader("Load a scenario")
    scenario_choice = st.selectbox(
        "Pre-built test scenarios",
        list(SCENARIO_FILES.keys()),
        index=0,
        help="Each scenario tests a different delivery environment"
    )

    st.markdown("**— or —**")
    uploaded = st.file_uploader("Upload your own CSV", type="csv")
    st.download_button("⬇ Download sample CSV", SAMPLE_CSV, "sample_stops.csv", "text/csv")

    st.divider()
    st.subheader("Depot location")
    depot_lat = st.number_input("Latitude",  value=47.6062, format="%.4f")
    depot_lng = st.number_input("Longitude", value=-122.3421, format="%.4f")

    st.divider()
    st.subheader("Simulation settings")
    shift_hrs = st.slider(
        "Shift length (hours)", 2, 10, 3, 1,
        help="How long the driver works. Shorter = more cascade visible."
    )
    n_sim = st.slider(
        "Monte Carlo scenarios", 100, 1000, 300, 100,
        help="How many random 'what-if' days to simulate for risk range"
    )

shift_mins = shift_hrs * 60.0

# ── CSV Format Guide ──────────────────────────────────────────────────────────

with st.expander("📖 How to format your CSV — click to expand", expanded=False):
    st.markdown("""
### What is a CSV?
A CSV (Comma-Separated Values) file is a simple spreadsheet you can create in Excel, Google Sheets, or even Notepad.
Each row = one delivery stop. Each column = one piece of information about that stop.

---

### Required columns

| Column | What it means | Example |
|--------|--------------|---------|
| `lat` | Latitude — the north/south GPS coordinate | `47.6062` |
| `lng` | Longitude — the east/west GPS coordinate | `-122.3321` |

> **Tip:** In Google Maps, right-click any location → the first number is `lat`, the second is `lng`.

---

### Optional columns (system uses smart defaults if missing)

| Column | What it means | Good values | Default if missing |
|--------|--------------|-------------|-------------------|
| `label` | Name of the stop — shown on the map | `"Amazon HQ"`, `"Stop 1"` | Auto-numbered |
| `dwell_mins` | How long the driver typically spends at this stop (minutes) | `3`–`10` min | `5.0 min` |
| `p_success` | Probability that someone is home / delivery succeeds (0 to 1) | `0.90` = 90% success rate | `0.85` |

---

### Column meanings in plain English

**`dwell_mins`** — "How long does this stop actually take?"
- A business mailroom: 2–3 min (someone at reception)
- A house: 4–6 min (ring doorbell, wait, leave note)
- An apartment with a locked lobby: 7–10 min (buzz intercom, wait for access)

**`p_success`** — "What's the chance the delivery actually goes through?"
- `0.95` = very reliable (office building, someone always there)
- `0.85` = typical residential
- `0.70` = risky (gated complex, elderly resident may not answer)

---

### Minimal valid example (2 columns only)
```
lat,lng
47.6062,-122.3321
47.6092,-122.3401
47.6001,-122.3291
```

### Full example (all columns)
```
lat,lng,label,dwell_mins,p_success
47.6062,-122.3321,Amazon HQ,3.0,0.97
47.6092,-122.3401,Capitol Hill Apt,7.0,0.82
47.6001,-122.3291,Beacon Hill House,5.0,0.88
```

---

### What the system does with your data
1. **Plans two routes** — Baseline (industry standard 2-min assumption) and Ours (data-driven P75 buffer)
2. **Simulates your actual shift** — checks which stops get cascaded when drivers run late
3. **Runs 300 random scenarios** — gives you a risk range (good day / bad day / average)
4. **Calculates savings** — translates missed stops into dollars per route per year
""")

# ── Load data ─────────────────────────────────────────────────────────────────

if uploaded is not None:
    df = pd.read_csv(uploaded)
    st.success(f"✅ Loaded your file: **{uploaded.name}** ({len(df)} stops)")
else:
    fname = SCENARIO_FILES[scenario_choice]
    if fname and (SCENARIOS_DIR / fname).exists():
        df = pd.read_csv(SCENARIOS_DIR / fname)
        st.info(f"📂 Loaded scenario: **{scenario_choice}** ({len(df)} stops)")
    else:
        df = pd.read_csv(io.StringIO(SAMPLE_CSV))
        st.info("📋 Using built-in 40-stop Seattle sample. Upload your CSV or pick a scenario above.")

required = {"lat", "lng"}
if not required.issubset(df.columns):
    st.error(f"❌ CSV must have `lat` and `lng` columns. Found: {list(df.columns)}")
    st.stop()

if "dwell_mins" not in df.columns: df["dwell_mins"] = 5.0
if "p_success"  not in df.columns: df["p_success"]  = 0.85
if "label"      not in df.columns: df["label"]      = [f"Stop {i+1}" for i in range(len(df))]

df = df.dropna(subset=["lat","lng"]).reset_index(drop=True)
n  = len(df)

if n < 3:
    st.warning("Need at least 3 stops to plan a route.")
    st.stop()

# ── Compute ───────────────────────────────────────────────────────────────────

coords = [(depot_lat, depot_lng)] + list(zip(df["lat"], df["lng"]))
with st.spinner("⏳ Building travel time matrix..."):
    try:
        mat = np.array(build_matrix(coords), dtype=float)
    except Exception:
        mat = build_haversine_matrix(coords)
np.fill_diagonal(mat, 0.0)

dwells    = df["dwell_mins"].tolist()
p_success = df["p_success"].tolist()
p75_dwell = float(np.percentile(dwells, 75))

nn_full     = nn_order(n, mat)
baseline_pl = plan_trim(nn_full, mat, PLANNED_DWELL_BASELINE, shift_mins)
ours_pl     = plan_trim(nn_full, mat, p75_dwell, shift_mins)

sim_b = simulate(baseline_pl, mat, dwells, p_success, shift_mins)
sim_o = simulate(ours_pl,     mat, dwells, p_success, shift_mins)

stops_mc = [
    {"aoi_id": -1, "aoi_type": 1, "package_count": 1, "hour_of_day": 10,
     "pred_mean": dwells[i], "pred_std": dwells[i]*0.4, "p_success": p_success[i]}
    for i in range(n)
]

class _DwellModel:
    def predict_with_std(self, *a, **kw): return 5.0, 3.0

with st.spinner(f"⏳ Running {n_sim} Monte Carlo scenarios..."):
    risk = simulate_route_risk(stops_mc, mat, _DwellModel(),
                               order=[i-1 for i in ours_pl], n_sim=n_sim,
                               shift_mins=shift_mins)

# ── Summary metrics ───────────────────────────────────────────────────────────

fadr_gain = (sim_o["fadr"] - sim_b["fadr"]) * 100
cascade_prevented = sim_b["cascade"] - sim_o.get("cascade", 0)
saving_per_route  = cascade_prevented * 12
saving_annual     = saving_per_route * 313 * 50

st.divider()
st.subheader("📊 Results at a glance")

c1, c2, c3, c4 = st.columns(4)
c1.metric(
    "Stops planned — industry standard",
    f"{len(baseline_pl)}/{n}",
    help="Plans assuming 2 min per stop (industry default). Overruns shift."
)
c2.metric(
    "Stops planned — our system",
    f"{len(ours_pl)}/{n}",
    help=f"Plans using realistic P75 dwell ({p75_dwell:.0f} min). Defers rest."
)
c3.metric(
    "Delivery success rate — standard",
    f"{sim_b['fadr']*100:.1f}%",
    help="FADR: % of all planned stops that get successfully delivered (cascade failures drag this down)"
)
c4.metric(
    "Delivery success rate — ours",
    f"{sim_o['fadr']*100:.1f}%",
    delta=f"{fadr_gain:+.1f}pp",
    help="FADR on our plan. Higher = more customers served, fewer complaints."
)

# Plain-English interpretation
if fadr_gain > 5:
    st.success(f"""
**What this means in plain English:**
The industry-standard approach assigns {sim_b['cascade']} stops your driver will physically never reach —
those customers get no delivery and no warning. Our system identifies this upfront, commits to {len(ours_pl)} stops,
and defers the rest with advance notification. Result: **{cascade_prevented} fewer surprise failures per route**,
{sim_o['fadr']*100:.0f}% vs {sim_b['fadr']*100:.0f}% delivery success rate.
""")
elif fadr_gain > 0:
    st.info(f"Modest improvement (+{fadr_gain:.1f}pp). Try the **Nightmare Route** scenario or reduce shift length for a more dramatic example.")
else:
    st.info("No cascade on this route — all stops fit in the shift. Try a shorter shift or more stops to see cascade prevention in action.")

st.divider()

# ── Map + Charts ──────────────────────────────────────────────────────────────

col_map, col_chart = st.columns([3, 2])

with col_map:
    st.subheader("🗺️ Route Map")
    st.caption("Green = our plan (realistic) · Orange dashed = standard plan · Click stops for details")

    center = [df["lat"].mean(), df["lng"].mean()]
    m = folium.Map(location=center, zoom_start=13, tiles="CartoDB positron")

    folium.Marker([depot_lat, depot_lng], popup="Depot (start/end)",
                  icon=folium.Icon(color="black", icon="home")).add_to(m)

    ours_coords = [(depot_lat, depot_lng)] + \
                  [(df.iloc[i-1]["lat"], df.iloc[i-1]["lng"]) for i in ours_pl]
    folium.PolyLine(ours_coords, color="green", weight=3, opacity=0.7,
                    tooltip="Our optimised route").add_to(m)

    base_coords = [(depot_lat, depot_lng)] + \
                  [(df.iloc[i-1]["lat"], df.iloc[i-1]["lng"]) for i in baseline_pl]
    folium.PolyLine(base_coords, color="orange", weight=2, opacity=0.5,
                    tooltip="Standard route", dash_array="6").add_to(m)

    ours_set = set(ours_pl)
    base_set = set(baseline_pl)
    for idx, row in df.iterrows():
        i_mat = idx + 1
        if i_mat in ours_set:
            color, icon_name = "green", "ok"
            tip = "In our plan"
        elif i_mat in base_set:
            color, icon_name = "orange", "minus"
            tip = "Standard plan only"
        else:
            color, icon_name = "gray", "remove"
            tip = "Not in either plan (too far)"
        folium.Marker(
            [row["lat"], row["lng"]],
            popup=(f"<b>{row['label']}</b><br>"
                   f"Dwell: {row['dwell_mins']} min<br>"
                   f"Success rate: {row['p_success']*100:.0f}%<br>"
                   f"Status: {tip}"),
            icon=folium.Icon(color=color, icon=icon_name),
        ).add_to(m)

    st_folium(m, width=700, height=450)
    st.caption("🟢 Our route &nbsp;|&nbsp; 🟠 Standard (dashed) &nbsp;|&nbsp; ⚫ Depot &nbsp;|&nbsp; ⚪ Not planned")

with col_chart:
    st.subheader("Delivery Success Rate")
    st.caption("First-Attempt Delivery Rate (FADR) — higher is better")

    fig = go.Figure()
    fig.add_bar(
        x=[f"Standard\n(2-min plan)", f"Our system\n(P75={p75_dwell:.0f}-min plan)"],
        y=[sim_b["fadr"]*100, sim_o["fadr"]*100],
        marker_color=["#f87171", "#22c55e"],
        text=[f"{sim_b['fadr']*100:.1f}%", f"{sim_o['fadr']*100:.1f}%"],
        textposition="outside",
    )
    fig.update_layout(
        yaxis_range=[0, 110], yaxis_title="% of stops successfully delivered",
        showlegend=False, height=280,
        plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
    )
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Risk Range (Our Plan)")
    st.caption("Monte Carlo simulation across 300 random days — what to expect on bad, average, and good days")

    r = risk
    fig2 = go.Figure()
    fig2.add_bar(
        x=["Worst 10%\n(bad day)", "Average day", "Best 10%\n(good day)"],
        y=[r.p10_fadr*100, r.mean_fadr*100, r.p90_fadr*100],
        marker_color=["#f87171", "#60a5fa", "#22c55e"],
        text=[f"{v:.0f}%" for v in [r.p10_fadr*100, r.mean_fadr*100, r.p90_fadr*100]],
        textposition="outside",
    )
    fig2.update_layout(
        title=f"Risk level: {r.risk_label}",
        yaxis_range=[0, 110], yaxis_title="Delivery success rate %",
        showlegend=False, height=250,
        plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
    )
    st.plotly_chart(fig2, use_container_width=True)

    # Financial impact
    st.subheader("💰 Financial Impact")
    col_a, col_b = st.columns(2)
    col_a.metric("Cascade failures prevented", f"{cascade_prevented} stops/route")
    col_b.metric("Cost saved per route", f"${saving_per_route}")
    st.metric(
        "Estimated annual saving (50 drivers, 313 days)",
        f"${saving_annual:,.0f}",
        help="$12 per cascade failure (redelivery cost + customer compensation)"
    )

# ── Glossary ──────────────────────────────────────────────────────────────────

with st.expander("📚 Glossary — what do these terms mean?"):
    st.markdown("""
| Term | Plain English |
|------|--------------|
| **FADR** | First-Attempt Delivery Rate — out of all stops planned today, what % actually got delivered? |
| **Cascade failure** | When a driver runs out of time mid-route. Every stop after the cutoff gets missed — with no warning to customers. Like a domino effect. |
| **Planned deferral** | Our system *proactively* removes stops it knows won't fit, and notifies those customers ahead of time. Better experience than a surprise miss. |
| **P75 dwell** | The 75th percentile of how long stops take. Planning to this buffer means only 25% of stops take longer than expected. |
| **Monte Carlo** | We simulate 300 different "possible days" by randomly varying dwell times. The result tells you: even on a bad day, what's the worst-case delivery rate? |
| **Risk: LOW / MEDIUM / HIGH** | How stable the delivery rate is across scenarios. LOW = consistently good. HIGH = highly variable. |
| **Depot** | The distribution centre where the driver starts and ends their shift. |
""")

st.divider()
st.subheader("📋 Stop Details")
st.caption("Full table of your stops with plan membership")
out_df = df[["label","lat","lng","dwell_mins","p_success"]].copy()
out_df["in_standard_plan"] = [(i+1) in set(baseline_pl) for i in range(n)]
out_df["in_our_plan"]      = [(i+1) in set(ours_pl)     for i in range(n)]
out_df.columns = ["Stop Name", "Lat", "Lng", "Dwell (min)", "Success Rate", "In Standard Plan", "In Our Plan"]
st.dataframe(out_df, use_container_width=True)
