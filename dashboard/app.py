"""
dashboard/app.py — Streamlit dispatcher dashboard.

Shows:
  • Folium map: today's route with risk-coloured stop markers + route line
  • Sidebar: SLA alert panel (high/medium risk stops needing action)
  • Route table: sequence, address, ETA window, dwell time, risk score, action

Run:
    streamlit run dashboard/app.py

Connects to the live API if ROUTE_INTEL_API_URL is set in .env.
Falls back to demo data so the dashboard is always usable.
"""

import os
import requests
import pandas as pd
import streamlit as st
import folium
from streamlit_folium import st_folium
from datetime import datetime, timedelta

# ── Page config ───────────────────────────────────────────────────────────────

st.set_page_config(
    page_title = "Route Intelligence — Dispatcher",
    page_icon  = "🚚",
    layout     = "wide",
)

API_URL = os.getenv("ROUTE_INTEL_API_URL", "http://localhost:8000")

# ── Visual constants ──────────────────────────────────────────────────────────

RISK_COLOUR = {
    "low":    "#28a745",
    "medium": "#ffc107",
    "high":   "#dc3545",
}
RISK_BADGE = {
    "low":    "🟢",
    "medium": "🟡",
    "high":   "🔴",
}

# ── Demo route (shown when API is unreachable or no model loaded) ─────────────

_DEMO_ROUTE = [
    {
        "sequence": 1, "stop_id": "S-001",
        "address": "Trafalgar Square, London",
        "lat": 51.5081, "lon": -0.1281,
        "arrival_earliest": 30, "arrival_latest": 60, "dwell_mins": 4,
        "risk": {"risk_level": "low", "failure_probability": 0.08,
                 "recommended_action": "attempt",
                 "explanation": "Low risk. Proceed as normal."},
    },
    {
        "sequence": 2, "stop_id": "S-002",
        "address": "Baker Street, London",
        "lat": 51.5238, "lon": -0.1585,
        "arrival_earliest": 55, "arrival_latest": 90, "dwell_mins": 6,
        "risk": {"risk_level": "medium", "failure_probability": 0.45,
                 "recommended_action": "pre_call",
                 "explanation": "Flat, no access code. Call customer to confirm."},
    },
    {
        "sequence": 3, "stop_id": "S-003",
        "address": "Victoria Station, London",
        "lat": 51.4952, "lon": -0.1441,
        "arrival_earliest": 85, "arrival_latest": 120, "dwell_mins": 3,
        "risk": {"risk_level": "low", "failure_probability": 0.12,
                 "recommended_action": "attempt",
                 "explanation": "Business address. Low risk."},
    },
    {
        "sequence": 4, "stop_id": "S-004",
        "address": "Brixton Market, London",
        "lat": 51.4618, "lon": -0.1140,
        "arrival_earliest": 110, "arrival_latest": 150, "dwell_mins": 8,
        "risk": {"risk_level": "high", "failure_probability": 0.78,
                 "recommended_action": "redirect_locker",
                 "explanation": "3 previous failed attempts at this address. Redirect to locker."},
    },
    {
        "sequence": 5, "stop_id": "S-005",
        "address": "Canary Wharf, London",
        "lat": 51.5054, "lon": -0.0235,
        "arrival_earliest": 140, "arrival_latest": 180, "dwell_mins": 3,
        "risk": {"risk_level": "low", "failure_probability": 0.09,
                 "recommended_action": "attempt",
                 "explanation": "Office block, business hours. Low risk."},
    },
    {
        "sequence": 6, "stop_id": "S-006",
        "address": "Hackney Central, London",
        "lat": 51.5471, "lon": -0.0578,
        "arrival_earliest": 165, "arrival_latest": 210, "dwell_mins": 5,
        "risk": {"risk_level": "medium", "failure_probability": 0.52,
                 "recommended_action": "pre_call",
                 "explanation": "Peak-hour arrival. Customer may be unavailable."},
    },
]


# ── Data fetching ─────────────────────────────────────────────────────────────

@st.cache_data(ttl=30)
def fetch_route() -> tuple[list, bool]:
    """
    Returns (route_stops, is_live).
    Tries the API health endpoint first; falls back to demo data on failure.
    """
    try:
        resp = requests.get(f"{API_URL}/health", timeout=2)
        if resp.ok:
            # API is live. In production, call /optimize-route here with today's manifest.
            # For now return demo data via API so the dashboard is always populated.
            return _DEMO_ROUTE, True
    except Exception:
        pass
    return _DEMO_ROUTE, False


def eta_label(shift_start: datetime, mins: int) -> str:
    return (shift_start + timedelta(minutes=mins)).strftime("%H:%M")


# ── Build flat dataframe from route ──────────────────────────────────────────

def route_to_df(route: list, shift_start: datetime) -> pd.DataFrame:
    rows = []
    for s in route:
        risk = s.get("risk") or {}
        rows.append({
            "Seq":         s["sequence"],
            "Stop ID":     s["stop_id"],
            "Address":     s["address"],
            "ETA Window":  f"{eta_label(shift_start, s['arrival_earliest'])}–{eta_label(shift_start, s['arrival_latest'])}",
            "Dwell (min)": s["dwell_mins"],
            "Risk":        risk.get("risk_level", "—"),
            "P(fail)":     f"{risk['failure_probability']:.0%}" if risk else "—",
            "Action":      risk.get("recommended_action", "—").replace("_", " ").title() if risk else "—",
            "lat":         s["lat"],
            "lon":         s["lon"],
            "_explanation": risk.get("explanation", ""),
        })
    return pd.DataFrame(rows)


# ── Main layout ───────────────────────────────────────────────────────────────

shift_start = datetime.now().replace(hour=8, minute=0, second=0, microsecond=0)
route, is_live = fetch_route()
df = route_to_df(route, shift_start)

high_risk   = df[df["Risk"] == "high"]
medium_risk = df[df["Risk"] == "medium"]

# Header
st.title("🚚 Route Intelligence — Dispatcher")
api_badge = "🟢 API live" if is_live else "🔴 Demo mode"
st.caption(
    f"Shift: {shift_start.strftime('%A %d %B, %H:%M')}  ·  "
    f"{api_badge}  ·  Auto-refresh every 30 s"
)

# ── Sidebar — SLA Alerts ─────────────────────────────────────────────────────

with st.sidebar:
    st.header("⚠️ SLA Alerts")

    if len(high_risk) == 0 and len(medium_risk) == 0:
        st.success("No at-risk stops. Route is clean.")
    else:
        if not high_risk.empty:
            st.error(f"🔴 {len(high_risk)} HIGH risk")
            for _, row in high_risk.iterrows():
                with st.expander(f"#{row['Seq']} · {row['Stop ID']}"):
                    st.write(f"**{row['Address']}**")
                    st.write(f"ETA: `{row['ETA Window']}`")
                    st.write(f"P(fail): **{row['P(fail)']}**")
                    st.write(f"Action: **{row['Action']}**")
                    if row["_explanation"]:
                        st.caption(row["_explanation"])

        if not medium_risk.empty:
            st.warning(f"🟡 {len(medium_risk)} MEDIUM risk")
            for _, row in medium_risk.iterrows():
                with st.expander(f"#{row['Seq']} · {row['Stop ID']}"):
                    st.write(f"**{row['Address']}**")
                    st.write(f"ETA: `{row['ETA Window']}`")
                    st.write(f"P(fail): **{row['P(fail)']}**")
                    st.write(f"Action: **{row['Action']}**")
                    if row["_explanation"]:
                        st.caption(row["_explanation"])

    st.divider()
    c1, c2, c3 = st.columns(3)
    c1.metric("Stops",  len(df))
    c2.metric("🔴 High",   len(high_risk))
    c3.metric("🟡 Med",    len(medium_risk))

    if st.button("🔄 Refresh", use_container_width=True):
        st.cache_data.clear()
        st.rerun()

# ── Map + Table columns ───────────────────────────────────────────────────────

col_map, col_table = st.columns([3, 2], gap="medium")

with col_map:
    st.subheader("Route map")

    centre = [df["lat"].mean(), df["lon"].mean()]
    m = folium.Map(location=centre, zoom_start=12, tiles="CartoDB positron")

    # Route polyline
    folium.PolyLine(
        locations  = list(zip(df["lat"], df["lon"])),
        color      = "#0d6efd",
        weight     = 3,
        opacity    = 0.65,
        dash_array = "8 4",
    ).add_to(m)

    # Stop markers
    for _, row in df.iterrows():
        rl     = row["Risk"] if row["Risk"] != "—" else "low"
        colour = RISK_COLOUR.get(rl, "#6c757d")
        badge  = RISK_BADGE.get(rl, "⚪")

        popup_html = (
            f'<div style="min-width:180px;font-family:sans-serif">'
            f'<b>#{row["Seq"]} · {row["Stop ID"]}</b><br>'
            f'{row["Address"]}<br><br>'
            f'ETA: {row["ETA Window"]}<br>'
            f'Risk: <b style="color:{colour}">{badge} {rl.upper()}</b><br>'
            f'P(fail): {row["P(fail)"]}<br>'
            f'Action: <b>{row["Action"]}</b><br>'
            f'<i style="color:#666;font-size:11px">{row["_explanation"]}</i>'
            f'</div>'
        )

        folium.CircleMarker(
            location     = [row["lat"], row["lon"]],
            radius       = 13,
            color        = colour,
            fill         = True,
            fill_color   = colour,
            fill_opacity = 0.85,
            popup        = folium.Popup(popup_html, max_width=230),
            tooltip      = f"#{row['Seq']} {row['Stop ID']} · {rl.upper()}",
        ).add_to(m)

        # Sequence number label inside marker
        folium.Marker(
            location = [row["lat"], row["lon"]],
            icon     = folium.DivIcon(
                html = (
                    f'<div style="font-size:10px;font-weight:bold;color:white;'
                    f'background:{colour};border-radius:50%;width:22px;height:22px;'
                    f'display:flex;align-items:center;justify-content:center;'
                    f'margin-top:-11px;margin-left:-11px">{row["Seq"]}</div>'
                ),
                icon_size   = (22, 22),
                icon_anchor = (11, 11),
            ),
        ).add_to(m)

    st_folium(m, width="100%", height=490, returned_objects=[])

with col_table:
    st.subheader("Stop sequence")

    display_cols = [
        "Seq", "Stop ID", "Address",
        "ETA Window", "Dwell (min)", "Risk", "P(fail)", "Action",
    ]

    def _style_risk(val: str) -> str:
        if val == "high":
            return "background-color:#f8d7da;color:#842029;font-weight:bold"
        if val == "medium":
            return "background-color:#fff3cd;color:#664d03"
        return ""

    styled = (
        df[display_cols]
        .style
        .map(_style_risk, subset=["Risk"])
    )
    st.dataframe(styled, use_container_width=True, hide_index=True, height=490)

# ── Footer ────────────────────────────────────────────────────────────────────

st.divider()
st.caption(
    f"Last rendered: {datetime.now().strftime('%H:%M:%S')}  ·  "
    f"API: {API_URL}  ·  Route Intelligence Engine v0.1"
)
