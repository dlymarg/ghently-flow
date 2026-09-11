import json
from google import genai
from google.genai import types
from sqlalchemy.orm import Session

from models import SensorNode
from physics import evaluate_sensor_telemetry, TriageLevel

# Initializes client using GEMINI_API_KEY from environment
client = genai.Client()


def build_system_prompt(sensor: SensorNode) -> str:
    """Injects spatial metadata and strict operational protocols into the system prompt."""
    return f"""You are the automated Crowd Safety AI Dispatcher for the City of Ghent command center.
Your task is to assess pedestrian crush risk at physical sensor locations using macroscopic fluid dynamics telemetry.

MONITORED NODE CONTEXT:
- Camera ID: {sensor.camera_id}
- Monitored Location: {sensor.sensor_placement}
- Coordinates: ({sensor.lat:.5f}, {sensor.lon:.5f})

CRITICAL PROTOCOLS:
1. You must immediately invoke 'evaluate_sensor_telemetry' using camera_id: '{sensor.camera_id}' to obtain ground-truth telemetry.
2. Read the returned JSON telemetry carefully:
   - If status is 'NOMINAL', output exactly:
     "STATUS GREEN: Operations normal at {sensor.sensor_placement}."
   - If status is 'STALE_TELEMETRY' or 'INSUFFICIENT_DATA', report an operational sensor degradation warning.
   - If status is 'ELEVATED' or 'CRITICAL', draft an urgent tactical crowd routing directive.
3. For routing directives:
   - Adopt an authoritative, professional incident-command tone.
   - Quote exact metrics from the tool output: Net Flux (Q_net), Acceleration (dQ/dt), and Current Inflow/Outflow.
   - Instruct ground staff to divert incoming pedestrian flow away from {sensor.sensor_placement} toward open zones (e.g., the open Korenmarkt square).
   - Keep final output structured, concise, and actionable.
"""


def run_triage_agent(session: Session, camera_id: str, model: str = "gemini-2.5-flash") -> str:
    """
    Executes a Gemini tool-calling ReAct triage loop grounded in physical continuity telemetry.
    """
    # 1. Retrieve spatial context from the database
    sensor = session.query(SensorNode).filter(SensorNode.camera_id == camera_id).first()
    if not sensor:
        return f"ERROR: Sensor node '{camera_id}' does not exist in the database."

    # 2. Define the tool as a scoped Python callable
    # Gemini automatically extracts the function schema from type hints and docstrings
    def evaluate_sensor_telemetry_tool(camera_id: str) -> str:
        """Calculates real-time macroscopic fluid continuity, volumetric flow rates,
        and fluid compression acceleration for a given camera node.

        Args:
            camera_id: Unique camera identifier (e.g., 'camera0', 'camera1').
        """
        metrics = evaluate_sensor_telemetry(session, camera_id)
        return json.dumps({
            "camera_id": metrics.camera_id,
            "timestamp": metrics.timestamp.isoformat(),
            "inflow_rate": metrics.inflow,
            "outflow_rate": metrics.outflow,
            "q_net_accumulation": metrics.q_net,
            "compression_acceleration_dq_dt": metrics.dq_dt,
            "delta_t_seconds": metrics.delta_t_seconds,
            "status": metrics.status.value,
            "deterministic_summary": metrics.directive_summary,
        })

    try:
        # 3. Generate response with automated tool execution
        response = client.models.generate_content(
            model=model,
            contents=f"Assess crowd safety and issue routing directives for {camera_id}.",
            config=types.GenerateContentConfig(
                system_instruction=build_system_prompt(sensor),
                tools=[evaluate_sensor_telemetry_tool],
                temperature=0.2,
            ),
        )

        return response.text or "STATUS NOMINAL: No corrective action required."

    except Exception as e:
        return (
            f"SYSTEM FAULT: Automated AI Triage Agent encountered an unhandled exception: {str(e)}. "
            f"Manual operator dispatch required for node '{camera_id}'."
        )
    