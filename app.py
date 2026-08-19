import streamlit as st
import plotly.express as px
from streamlit_folium import st_folium
import folium

st.set_page_config(layout="wide", page_title="Ghent Crowd Safety Command Center")
st.title("Welcome to the Ghent Crowd Safety Command Center!")

# Sidebar selector
camera_id = st.sidebar.selectbox("Select Camera Node", ["camera0", "camera1"])

# Fetch metrics from DB session
# (Assuming session & query helpers are initialized)
latest_log = get_latest_log(session, camera_id) 

# Top Section: KPI Metrics
col1, col2, col3 = st.columns(3)
col1.metric("Monitored Access Points", "2")
col2.metric("Current Inflow Rate", f"{latest_log.current_inflow:.0f} ped/min")
col3.metric("Current Outflow Rate", f"{latest_log.current_outflow:.0f} ped/min")

# Middle Section: Interactive Map + Charts
col_map, col_chart = st.columns([1, 1])

with col_map:
    m = folium.Map(location=[51.05394, 3.72311], zoom_start=16)
    folium.Marker([51.05394, 3.72311], popup="camera0 - Sint-Niklaaskerk West").add_to(m)
    folium.Marker([51.05410, 3.72330], popup="camera1 - Sint-Niklaaskerk North").add_to(m)
    st_folium(m, width=500, height=350)

with col_chart:
    df_logs = get_historical_logs(session, camera_id)
    fig = px.line(df_logs, x="created_at", y=["current_inflow", "current_outflow"], title="Telemetry History")
    st.plotly_chart(fig, use_container_width=True)

# Bottom Section: LLM Safety Output
st.subheader("AI Directive Output")
if st.button("Generate AI Safety Triage Directive"):
    with st.spinner("Executing ReAct Tool Call..."):
        ai_output = run_triage_agent(session, camera_id)
        st.info(ai_output)