from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Optional
from sqlalchemy.orm import Session
from models import CrowdChokepoint, SensorNode


class TriageLevel(str, Enum):
    NOMINAL = "NOMINAL"
    ELEVATED = "ELEVATED"
    CRITICAL = "CRITICAL"
    STALE = "STALE_TELEMETRY"
    INSUFFICIENT_DATA = "INSUFFICIENT_DATA"


@dataclass(frozen=True)
class FluidStrainMetrics:
    camera_id: str
    timestamp: datetime
    inflow: float
    outflow: float
    q_net: float
    dq_dt: float
    delta_t_seconds: float
    status: TriageLevel
    directive_summary: str


def compute_fluid_continuity(
    latest: CrowdChokepoint,
    previous: CrowdChokepoint,
    max_staleness_seconds: float = 600.0,
    critical_q_threshold: float = 40.0,
    critical_accel_threshold: float = 10.0,
) -> FluidStrainMetrics:
    """
    Evaluates mass conservation and time-normalized compression acceleration.
    - Q_net = Inflow - Outflow
    - dQ/dt = (Q_net_latest - Q_net_prev) / delta_t
    """
    # Ensure created_at is converted to datetime if stored as string in SQLite
    t_latest = (
        datetime.fromisoformat(latest.created_at)
        if isinstance(latest.created_at, str)
        else latest.created_at
    )
    t_prev = (
        datetime.fromisoformat(previous.created_at)
        if isinstance(previous.created_at, str)
        else previous.created_at
    )

    delta_t = (t_latest - t_prev).total_seconds()

    # Guard 1: Non-positive time delta
    if delta_t <= 0:
        delta_t = 60.0  # Fallback default interval (1 minute)

    # Fluid Flux: Q_net = Inflow - Outflow
    q_in = float(latest.current_inflow or 0.0)
    q_out = float(latest.current_outflow or 0.0)
    q_net_latest = q_in - q_out

    q_in_prev = float(previous.current_inflow or 0.0)
    q_out_prev = float(previous.current_outflow or 0.0)
    q_net_prev = q_in_prev - q_out_prev

    # Time-normalized acceleration (pedestrians / min²)
    delta_t_minutes = delta_t / 60.0
    dq_dt = (q_net_latest - q_net_prev) / delta_t_minutes

    # Decision Matrix
    if q_net_latest > critical_q_threshold and dq_dt > critical_accel_threshold:
        status = TriageLevel.CRITICAL
        summary = (
            f"CRITICAL CRUSH WARNING: Net accumulation is +{q_net_latest:.0f} ped/min "
            f"with acceleration of +{dq_dt:.1f} ped/min². Immediate crowd diversion required."
        )
    elif q_net_latest > (critical_q_threshold * 0.375):  # Equivalent to > 15
        status = TriageLevel.ELEVATED
        summary = (
            f"ELEVATED FLUID CONGESTION: Net accumulation is +{q_net_latest:.0f} ped/min. "
            f"Downstream dissipation is decelerating."
        )
    else:
        status = TriageLevel.NOMINAL
        summary = (
            f"STABLE COHESIVE STREAM: Inflow and outflow rates nominal. "
            f"Net flux: {q_net_latest:+.0f} ped/min, acceleration: {dq_dt:+.1f} ped/min²."
        )

    return FluidStrainMetrics(
        camera_id=latest.camera_id,
        timestamp=t_latest,
        inflow=q_in,
        outflow=q_out,
        q_net=q_net_latest,
        dq_dt=dq_dt,
        delta_t_seconds=delta_t,
        status=status,
        directive_summary=summary,
    )


def evaluate_sensor_telemetry(session: Session, camera_id: str) -> FluidStrainMetrics:
    """
    Data access layer: Retrieves the 2 most recent frames descending by timestamp
    and executes fluid continuity calculations.
    """
    records = (
        session.query(CrowdChokepoint)
        .filter(CrowdChokepoint.camera_id == camera_id)
        .order_by(CrowdChokepoint.created_at.desc())
        .limit(2)
        .all()
    )

    if len(records) < 2:
        now = datetime.utcnow()
        return FluidStrainMetrics(
            camera_id=camera_id,
            timestamp=now,
            inflow=records[0].current_inflow if records else 0.0,
            outflow=records[0].current_outflow if records else 0.0,
            q_net=0.0,
            dq_dt=0.0,
            delta_t_seconds=0.0,
            status=TriageLevel.INSUFFICIENT_DATA,
            directive_summary="INITIALIZING SYSTEM: Insufficient sequential frames for continuity calculations.",
        )

    return compute_fluid_continuity(latest=records[0], previous=records[1])
