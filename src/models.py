from datetime import datetime
from typing import Tuple
from sqlalchemy import (
    Column,
    String,
    Float,
    DateTime,
    ForeignKey,
    Index,
    create_engine
)
from sqlalchemy.orm import declarative_base, relationship

Base = declarative_base()

class SensorNode(Base):
    """
    Mirrors the 'Sensor Node' Object Type in Palantir Foundry.
    Represents physical optical sensors monitoring choke points.
    """
    __tablename__ = "sensor_nodes"

    # Primary Key
    camera_id = Column(String(50), primary_key=True, index=True)
    
    # Metadata & Spatial Properties
    sensor_placement = Column(String(255), nullable=False)
    lat = Column(Float, nullable=False)
    lon = Column(Float, nullable=False)

    # 1-to-Many Dynamic Relationship
    # lazy='dynamic' returns a Query object rather than loading all objects into RAM.
    crowd_chokepoints = relationship(
        "CrowdChokepoint",
        back_populates="sensor",
        cascade="all, delete-orphan",
        lazy="dynamic"
    )

    @property
    def location(self) -> Tuple[float, float]:
        """Replicates the Foundry Geopoint base type interface."""
        return (self.lat, self.lon)

    def __repr__(self) -> str:
        return f"<SensorNode(camera_id='{self.camera_id}', placement='{self.sensor_placement}')>"


class CrowdChokepoint(Base):
    """
    Mirrors the 'Crowd Chokepoint' Object Type in Palantir Foundry.
    Contains time-series telemetry logs of crowd flux and compression.
    """
    __tablename__ = "crowd_chokepoints"

    # Composite Primary Key: camera_id + ISO timestamp
    chokepoint_pk = Column(String(100), primary_key=True)
    
    # Foreign Key Linking to Parent Sensor
    camera_id = Column(String(50), ForeignKey("sensor_nodes.camera_id", ondelete="CASCADE"), nullable=False)
    
    # Telemetry Attributes
    created_at = Column(DateTime, nullable=False)
    current_inflow = Column(Float, nullable=False, default=0.0)
    current_outflow = Column(Float, nullable=False, default=0.0)
    
    # Pipeline Builder Output Columns
    risk_level = Column(String(20), nullable=False, default="NOMINAL")
    safety_directive = Column(String(500), nullable=True, default="")

    # Relationship Back to Sensor Node
    sensor = relationship("SensorNode", back_populates="crowd_chokepoints")

    # Table Constraints & Composite Indexes
    __table_args__ = (
        Index("ix_camera_created_desc", "camera_id", created_at.desc()),
    )

    def __repr__(self) -> str:
        return f"<CrowdChokepoint(pk='{self.chokepoint_pk}', in={self.current_inflow}, out={self.current_outflow})>"


def init_db(database_url: str = "sqlite:///data/ghent_telemetry.db"):
    """Creates the SQLite database engine and generates tables with compound indexes."""
    engine = create_engine(database_url, echo=False)
    Base.metadata.create_all(engine)
    return engine
