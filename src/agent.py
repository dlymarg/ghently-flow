import json
from dataclasses import asdict
from typing import Optional
from openai import OpenAI
from sqlalchemy.orm import Session

from models import SensorNode
from physics import evaluate_sensor_telemetry, TriageLevel

client = OpenAI()

# 1. Define strict Tool Calling Schema
TRIAGE_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "evaluate_sensor_telemetry",
            "description": "Calculates real-time macroscopic fluid continuity, volumetric flow rates, and fluid compression acceleration for a given camera node.",
            "parameters": {
                "type": "object",
                "properties": {
                    "camera_id": {
                        "type": "string",
                        "description": "Unique camera identifier (e.g., 'camera0', 'camera1')",
                    }
                },
                "required": ["camera_id"],
            },
        },
    }
]


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


def run_triage_agent(session: Session, camera_id: str, model: str = "gpt-4o") -> str:
    """
    Executes a ReAct tool-calling loop grounded in physical continuity telemetry.
    """
    # 1. Retrieve spatial context from the database
    sensor = session.query(SensorNode).filter(SensorNode.camera_id == camera_id).first()
    if not sensor:
        return f"ERROR: Sensor node '{camera_id}' does not exist in the database."

    messages = [
        {"role": "system", "content": build_system_prompt(sensor)},
        {"role": "user", "content": f"Assess crowd safety and issue routing directives for {camera_id}."},
    ]

    try:
        # Initial model completion request
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            tools=TRIAGE_TOOLS,
            tool_choice="auto",
            temperature=0.0,  # Deterministic output for emergency operations
        )

        response_message = response.choices[0].message

        # 2. Safely verify if the model invoked a tool call
        if not response_message.tool_calls:
            return response_message.content or "STATUS NOMINAL: No corrective action required."

        # Append model's tool call intention to conversation history
        messages.append(response_message)

        # 3. Execute tool calls safely
        for tool_call in response_message.tool_calls:
            if tool_call.function.name == "evaluate_sensor_telemetry":
                args = json.loads(tool_call.function.arguments)
                target_camera = args.get("camera_id", camera_id)

                # Execute deterministic Python physics calculation
                metrics = evaluate_sensor_telemetry(session, target_camera)

                # Serialize dataclass to JSON string for the tool response
                metrics_payload = json.dumps(
                    {
                        "camera_id": metrics.camera_id,
                        "timestamp": metrics.timestamp.isoformat(),
                        "inflow_rate": metrics.inflow,
                        "outflow_rate": metrics.outflow,
                        "q_net_accumulation": metrics.q_net,
                        "compression_acceleration_dq_dt": metrics.dq_dt,
                        "delta_t_seconds": metrics.delta_t_seconds,
                        "status": metrics.status.value,
                        "deterministic_summary": metrics.directive_summary,
                    }
                )

                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "content": metrics_payload,
                    }
                )

        # 4. Final synthesis step: LLM interprets the physics payload into command directives
        final_response = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=0.2,
        )

        return final_response.choices[0].message.content or "No directive generated."

    except Exception as e:
        # Fail-safe command center message
        return (
            f"SYSTEM FAULT: Automated AI Triage Agent encountered an unhandled exception: {str(e)}. "
            f"Manual operator dispatch required for node '{camera_id}'."
        )
    