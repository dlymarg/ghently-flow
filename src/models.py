from sqlalchemy import create_engine, Column, String, Float, DateTime, ForeignKey
from sqlalchemy.orm import declarative_base, relationship

Base = declarative_base()

class SensorNode(Base):
    __tablename__ = 'sensor_nodes'
    
    camera_id = Column(String, primary_key=True)
    lat = Column(Float, nullable=False)
    lon = Column(Float, nullable=False)
    sensor_placement = Column(String)
    
    # 1-to-Many Relationship Link
    chokepoints = relationship("CrowdChokepoint", back_populates="sensor")

class CrowdChokepoint(Base):
    __tablename__ = 'crowd_chokepoints'
    
    chokepoint_pk = Column(String, primary_key=True)
    camera_id = Column(String, ForeignKey('sensor_nodes.camera_id'))
    created_at = Column(DateTime, nullable=False)
    current_inflow = Column(Float)
    current_outflow = Column(Float)
    
    sensor = relationship("SensorNode", back_populates="chokepoints")