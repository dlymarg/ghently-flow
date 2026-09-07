import pandas as pd
import re
import sqlite3
from pathlib import Path

def parse_wkt_point(wkt_str: str):
    """
    Extracts (latitude, longitude) from WKT string: 'POINT (3.72311 51.05394)'
    WKT standard uses POINT (Longitude/X Latitude/Y).
    """
    if not isinstance(wkt_str, str):
        return None, None
    matches = re.findall(r"[-+]?\d*\.\d+|\d+", wkt_str)
    if len(matches) >= 2:
        val1, val2 = float(matches[0]), float(matches[1])
        # Ghent coordinates: Lat ~51.05, Lon ~3.72
        lat = val1 if val1 > 10 else val2
        lon = val2 if val1 > 10 else val1
        return lat, lon
    return None, None

def run_etl(telemetry_csv: str, camera_locations_csv: str, db_path: str):
    # ==========================================
    # Branch 1: Sensor Nodes (camera_locations)
    # ==========================================
    df_cameras = pd.read_csv(camera_locations_csv)
    
    # Add GeoPoint columns from WKT
    coords = df_cameras['location'].apply(parse_wkt_point)
    df_cameras['lat'] = [c[0] for c in coords]
    df_cameras['lon'] = [c[1] for c in coords]
    
    # Normalize column names for the ORM
    df_cameras = df_cameras.rename(columns={
        'Camera Id': 'camera_id',
        'Sensor Placement': 'sensor_placement'
    })
    
    # ==========================================
    # Branch 2: Telemetry (druktemeting...)
    # ==========================================
    df_raw = pd.read_csv(telemetry_csv)
    
    # 1. Unpivot columns: identify identifier cols vs metric measurement cols
    id_cols = [col for col in ['timestamp', 'createdat', 'time'] if col in df_raw.columns]
    time_col = id_cols[0] if id_cols else df_raw.columns[0]
    metric_cols = [c for c in df_raw.columns if c != time_col]
    
    df_unpivoted = df_raw.melt(
        id_vars=[time_col],
        value_vars=metric_cols,
        var_name='measurement_stream',
        value_name='pedestrian_count'
    )
    
    # 2. Apply regex to extract camera_id and direction (in/out)
    # E.g., 'camera0_inflow' -> camera_id='camera0', direction='inflow'
    df_unpivoted['camera_id'] = df_unpivoted['measurement_stream'].str.extract(r'(camera\d+|sensor\d+)', flags=re.IGNORECASE)[0].str.lower()
    df_unpivoted['direction'] = df_unpivoted['measurement_stream'].apply(
        lambda x: 'current_inflow' if 'in' in str(x).lower() else ('current_outflow' if 'out' in str(x).lower() else 'other')
    )
    
    # 3. Drop unneeded raw stream column
    df_filtered = df_unpivoted[df_unpivoted['direction'].isin(['current_inflow', 'current_outflow'])].copy()
    
    # 4. Aggregate inflow + outflow side-by-side
    df_agg = df_filtered.pivot_table(
        index=[time_col, 'camera_id'],
        columns='direction',
        values='pedestrian_count',
        aggfunc='sum'
    ).reset_index()
    
    df_agg = df_agg.rename(columns={time_col: 'created_at'})
    df_agg['created_at'] = pd.to_datetime(df_agg['created_at'])
    df_agg['current_inflow'] = df_agg.get('current_inflow', 0.0).fillna(0.0)
    df_agg['current_outflow'] = df_agg.get('current_outflow', 0.0).fillna(0.0)
    
    # 5. Create composite Primary Key
    df_agg['chokepoint_pk'] = df_agg['camera_id'] + "_" + df_agg['created_at'].dt.strftime('%Y-%m-%dT%H:%M:%S.000Z')
    
    # 6. Create risk_level & placeholder columns
    net_flux = df_agg['current_inflow'] - df_agg['current_outflow']
    df_agg['risk_level'] = net_flux.apply(lambda q: 'CRITICAL' if q > 40 else ('ELEVATED' if q > 15 else 'NOMINAL'))
    df_agg['safety_directive'] = ""

    # ==========================================
    # Database Persistence: Two Relational Tables
    # ==========================================
    conn = sqlite3.connect(db_path)
    
    # Table 1: Sensor Nodes (Parent)
    df_cameras[['camera_id', 'sensor_placement', 'lat', 'lon']].to_sql(
        'sensor_nodes', conn, if_exists='replace', index=False
    )
    
    # Table 2: Crowd Chokepoints (Child logs with Foreign Key camera_id)
    df_agg[['chokepoint_pk', 'camera_id', 'created_at', 'current_inflow', 'current_outflow', 'risk_level', 'safety_directive']].to_sql(
        'crowd_chokepoints', conn, if_exists='replace', index=False
    )
    
    # Add indexes for high-speed chronological queries
    cursor = conn.cursor()
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_camera_time ON crowd_chokepoints(camera_id, created_at DESC);")
    conn.commit()
    conn.close()
    
    print(f"ETL completed successfully: {len(df_cameras)} sensors and {len(df_agg)} chokepoint logs loaded into {db_path}.")

if __name__ == "__main__":
    run_etl(
        telemetry_csv="data/druktemeting-gent-kortedagsteeg.csv",
        camera_locations_csv="data/camera_locations.csv",
        db_path="data/ghent_telemetry.db"
    )