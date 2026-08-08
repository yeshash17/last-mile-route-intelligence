import streamlit as st

st.set_page_config(
    page_title="Last Mile Route Intelligence",
    page_icon="🚚",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("🚚 Last Mile Route Intelligence Engine")
st.markdown("""
A decision intelligence system for last-mile delivery optimization.

**Built on:**
- Amazon Last Mile 2021 dataset (6,112 US routes)
- LaDe dataset (5 Chinese cities, 5.5M deliveries)
- OSRM real road routing (WA · IL · TX · MA)

---

### Three pages:

| Page | What it shows |
|---|---|
| **Route Planner** | Upload your stops → optimized plan + FADR risk score |
| **Cascade Demo** | Live animation: how dwell underestimation causes cascade failures |
| **Benchmarks** | Validated results vs Amazon professional drivers |

Use the sidebar to navigate.
""")

st.info("👈 Select a page from the sidebar to get started.")
