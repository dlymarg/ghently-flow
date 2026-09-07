import streamlit as st
import pandas as pd
import plotly.express as px
import folium
from streamlit_folium import st_folium
from sqlalchemy.orm import sessionmaker, Session

from models import init_db, SensorNode, CrowdChokepoint
from physics import evaluate_sensor_telemetry, TriageLevel
from agent import run_triage_agent

# ---------------------------------------------------------
# Page Configuration & Database Resource Management
# ---------------------------------------------------------
st.set_page_config(
    page_title="Ghent Crowd Safety Command Center",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

@st.cache_resource
def get_db_session_factory():
    """Initializes and caches the database connection pool across reruns."""
    engine = init_db("sqlite:///data/ghent_telemetry.db")
    return sessionmaker(bind=engine)

SessionFactory = get_db_session_factory()

def get_db() -> Session:
    return SessionFactory()

# ---------------------------------------------------------
# Session State Initialization (Prevents UI Reset Bugs)
# ---------------------------------------------------------
if "ai_directive" not in st.session_state:
    st.session_state.ai_directive = None
if "selected_camera_id" not in st.session_state:
    st.session_state.selected_camera_id = None

# ---------------------------------------------------------
# Sidebar: Dynamic Node Selection & Ingestion Metadata
# ---------------------------------------------------------
with get_db() as db:
    sensors = db.query(SensorNode).order_by(SensorNode.camera_id.asc()).all()
    sensor_map = {s.camera_id: s for s in sensors}

if not sensors:
    st.error("DATABASE EMPTY: Run 'python src/etl.py' to ingest telemetry before launching.")
    st.stop()

st.sidebar.title("🛡️ Ghent Command Center")
st.sidebar.markdown("**Urban Pedestrian Flow Monitoring**")

camera_options = list(sensor_map.keys())
selected_camera_id = st.sidebar.selectbox(
    "Select Monitored Access Node",
    options=camera_options,
    format_func=lambda cid: f"{cid.upper()} - {sensor_map[cid].sensor_placement}"
)

# Clear cached AI output if the operator switches cameras
if st.session_state.selected_camera_id != selected_camera_id:
    st.session_state.selected_camera_id = selected_camera_id
    st.session_state.ai_directive = None

active_sensor = sensor_map[selected_camera_id]
st.sidebar.divider()
st.sidebar.markdown(f"**Coordinates:** `{active_sensor.lat:.5f}, {active_sensor.lon:.5f}`")
st.sidebar.markdown(f"**Placement:** {active_sensor.sensor_placement}")

# ---------------------------------------------------------
# Core Telemetry & Real-Time Physics Engine Evaluation
# ---------------------------------------------------------
with get_db() as db:
    metrics = evaluate_sensor_telemetry(db, selected_camera_id)

# Top Banner: Operational Phase Transition Status
status_color_map = {
    TriageLevel.NOMINAL: ("green", "STATUS NOMINAL: Flow dynamics operating within safe continuum limits."),
    TriageLevel.ELEVATED: ("orange", "STATUS ELEVATED: Fluid density increasing. Downstream dissipation decelerating."),
    TriageLevel.CRITICAL: ("red", "CRITICAL CRUSH WARNING: Severe crowd compression detected. Action required."),
    TriageLevel.STALE: ("gray", "TELEMETRY DEGRADED: Signal latency exceeds operational freshness limits."),
    TriageLevel.INSUFFICIENT_DATA: ("gray", "INITIALIZING: Insufficient sequential frames for continuity calculations.")
}
badge_color, status_text = status_color_map.get(metrics.status, ("gray", "UNKNOWN STATE"))

st.title("Ghent Crowd Safety Command Center")
st.caption(f"Live Node: Sint-Niklaaskerk ({active_sensor.sensor_placement}) | Telemetry Frame: {metrics.timestamp.strftime('%Y-%m-%d %H:%M:%S')}")

if metrics.status == TriageLevel.CRITICAL:
    st.error(f"🚨 **{status_text}**")
elif metrics.status == TriageLevel.ELEVATED:
    st.warning(f"⚠️ **{status_text}**")
else:
    st.success(f"✅ **{status_text}**")

# ---------------------------------------------------------
# KPI Metrics Cards (Fluid Dynamics & Flux Rates)
# ---------------------------------------------------------
col1, col2, col3, col4, col5 = st.columns(5)
col1.metric("Active Sensor Nodes", len(sensors))
col2.metric("Inflow Rate ($Q_{in}$)", f"{metrics.inflow:.0f} ped/min")
col3.metric("Outflow Rate ($Q_{out}$)", f"{metrics.outflow:.0f} ped/min")

# Delta indicators show positive accumulation in red, clearing in green
col4.metric(
    "Net Flux ($Q_{net}$)",
    f"{metrics.q_net:+.0f} ped/min",
    delta=f"{metrics.q_net:.0f} accumulation",
    delta_color="inverse"
)
col5.metric(
    "Compression Accel ($dQ/dt$)",
    f"{metrics.dq_dt:+.1f} ped/min²",
    delta=f"{metrics.dq_dt:.1f} acceleration",
    delta_color="inverse"
)

st.divider()

# ---------------------------------------------------------
# Geospatial Map & Real-Time Telemetry Trend
# ---------------------------------------------------------
col_map, col_chart = st.columns([1, 1])

with col_map:
    st.subheader("Geospatial Digital Twin")
    # Centered over Sint-Niklaaskerk / active node
    ghent_map = folium.Map(location=[active_sensor.lat, active_sensor.lon], zoom_start=17, tiles="CartoDB positron")

    for cid, s in sensor_map.items():
        is_selected = (cid == selected_camera_id)
        pin_color = "red" if (is_selected and metrics.status == TriageLevel.CRITICAL) else ("blue" if is_selected else "gray")
        
        folium.Marker(
            location=[s.lat, s.lon],
            tooltip=f"{cid.upper()}: {s.sensor_placement}",
            popup=f"<b>{cid}</b><br>{s.sensor_placement}",
            icon=folium.Icon(color=pin_color, icon="camera", prefix="fa")
        ).add_to(ghent_map)

    st_folium(ghent_map, width="100%", height=380, returned_objects=[])

with col_chart:
    st.subheader("Temporal Flux Continuum")
    with get_db() as db:
        # Window query to the latest 100 frames sorted chronologically ascending for Plotly
        recent_logs = (
            db.query(CrowdChokepoint)
            .filter(CrowdChokepoint.camera_id == selected_camera_id)
            .order_by(CrowdChokepoint.created_at.desc())
            .limit(100)
            .all()
        )
        recent_logs.reverse()

    if recent_logs:
        df_plot = pd.DataFrame([
            {
                "Timestamp": log.created_at,
                "Inflow": log.current_inflow,
                "Outflow": log.current_outflow,
                "Net Accumulation": log.current_inflow - log.current_outflow
            }
            for log in recent_logs
        ])
        fig = px.line(
            df_plot,
            x="Timestamp",
            y=["Inflow", "Outflow", "Net Accumulation"],
            color_discrete_map={"Inflow": "#d9534f", "Outflow": "#5cb85c", "Net Accumulation": "#337ab7"},
            labels={"value": "Pedestrians / Minute", "variable": "Stream"}
        )
        fig.update_layout(margin=dict(l=10, r=10, t=25, b=10), height=380, legend=dict(orientation="h", y=1.1))
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No time-series telemetry available for this node.")

st.divider()

# ---------------------------------------------------------
# ReAct AI Dispatch & Triage Directive
# ---------------------------------------------------------
st.subheader("Autonomous AI Triage Dispatcher")
col_btn, col_blank = st.columns([1, 4])

with col_btn:
    generate_clicked = st.button("Generate Tactical Directive", type="primary")

if generate_clicked:
    with st.spinner("Invoking ReAct Agent & evaluating fluid continuity tool..."):
        with get_db() as db:
            st.session_state.ai_directive = run_triage_agent(db, selected_camera_id)

if st.session_state.ai_directive:
    st.markdown(
        f"""
        <div style="background-color: #f8f9fa; border-left: 5px solid {badge_color}; padding: 15px; border-radius: 4px; color: #212529;">
            <strong>AI Command Directive Output:</strong><br>
            {st.session_state.ai_directive}
        </div>
        """,
        unsafe_allow_html=True
    )
    