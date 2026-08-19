from sqlalchemy.orm import Session
from models import CrowdChokepoint

def evaluate_macroscopic_fluid_strain(session: Session, camera_id: str) -> str:
    # Query database for the 2 most recent frames descending by timestamp
    latest_two_logs = (
        session.query(CrowdChokepoint)
        .filter(CrowdChokepoint.camera_id == camera_id)
        .order_by(CrowdChokepoint.created_at.desc())
        .limit(2)
        .all()
    )
    
    if len(latest_two_logs) < 2:
        return "INITIALIZING SYSTEM - INSUFFICIENT TEMPORAL DATA"
        
    latest_frame, previous_frame = latest_two_logs[0], latest_two_logs[1]
    
    q_in = latest_frame.current_inflow or 0.0
    q_out = latest_frame.current_outflow or 0.0
    
    # Mass conservation equation: Q_net = Inflow - Outflow
    q_net_latest = q_in - q_out
    q_net_prev = (previous_frame.current_inflow or 0.0) - (previous_frame.current_outflow or 0.0)
    dq_dt = q_net_latest - q_net_prev
    
    if q_net_latest > 40 and dq_dt > 10:
        return f"CRITICAL CRUSH WARNING: Net accumulation is +{q_net_latest:.0f} ped/min (acceleration: +{dq_dt:.0f}/min²). Redirect crowd immediately."
    elif q_net_latest > 15:
        return f"ELEVATED FLUID CONGESTION: Net accumulation is +{q_net_latest:.0f} ped/min."
    else:
        return f"STABLE COHESIVE STREAM: Inflow and Outflow rates operating within safe boundaries. Net flux is {q_net_latest:.0f} ped/min."