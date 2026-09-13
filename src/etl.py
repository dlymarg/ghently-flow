import os
import re
import sqlite3
from typing import Any, Optional, Tuple
import pandas as pd


def parse_geo_coordinates(coord_val: Any) -> Tuple[Optional[float], Optional[float]]:
    """
    Extracts (latitude, longitude) floats from multiple geographic formats:
      - OpenDataSoft geo_point_2d: '51.05394, 3.72311' (Lat, Lon)
      - WKT Point strings: 'POINT (3.72311 51.05394)' (Lon, Lat)
      - JSON / Python lists or tuples: [51.05394, 3.72311]
      - Dictionaries: {'lat': 51.05394, 'lon': 3.72311}
    """
    if coord_val is None or (isinstance(coord_val, float) and pd.isna(coord_val)):
        return None, None

    if isinstance(coord_val, dict):
        lat = coord_val.get("lat") or coord_val.get("latitude")
        lon = coord_val.get("lon") or coord_val.get("lng") or coord_val.get("longitude")
        if lat is not None and lon is not None:
            return float(lat), float(lon)

    if isinstance(coord_val, (list, tuple)) and len(coord_val) >= 2:
        val1, val2 = float(coord_val[0]), float(coord_val[1])
    else:
        val_str = str(coord_val).strip()
        matches = re.findall(r"[-+]?\d*\.?\d+", val_str)
        if len(matches) < 2:
            return None, None
        val1, val2 = float(matches[0]), float(matches[1])

    # Ghent geographic bounds heuristic (Latitude ~51.05°N, Longitude ~3.72°E)
    if val1 > 10.0 and val2 < 10.0:
        return val1, val2
    if val2 > 10.0 and val1 < 10.0:
        return val2, val1

    # OGC WKT standard fallback: POINT (lon lat)
    if "POINT" in str(coord_val).upper():
        return val2, val1

    return val1, val2


def load_csv_auto_delim(filepath: str) -> pd.DataFrame:
    """Reads CSV while detecting comma vs semicolon delimiters automatically."""
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Input file not found at: {filepath}")

    with open(filepath, "r", encoding="utf-8") as f:
        first_line = f.readline()

    delimiter = ";" if ";" in first_line and "," not in first_line else ","
    return pd.read_csv(filepath, sep=delimiter)


def run_etl(telemetry_csv: str, camera_locations_csv: str, db_path: str):
    """
    Executes full ETL pipeline:
      1. Normalizes camera metadata and parses spatial coordinates into lat/lon floats.
      2. Unpivots 43-column telemetry down to line-level pedestrian flux records.
      3. Aggregates directional inflow/outflow per node per timestamp.
      4. Builds composite primary keys and pre-computes operational risk thresholds.
      5. Generates indexed SQLite tables ('sensor_nodes' and 'crowd_chokepoints').
    """
    # -------------------------------------------------------------
    # 1. Process Sensor Metadata (camera_locations)
    # -------------------------------------------------------------
    df_cameras = load_csv_auto_delim(camera_locations_csv)
    df_cameras.columns = df_cameras.columns.str.strip()

    # Standardize primary identification and placement columns
    col_mapping = {
        "Camera Id": "camera_id",
        "camera_id": "camera_id",
        "Sensor Placement": "sensor_placement",
        "sensor_placement": "sensor_placement",
    }
    df_cameras = df_cameras.rename(columns=col_mapping)

    if "camera_id" not in df_cameras.columns:
        raise KeyError(f"Missing 'camera_id' column in {camera_locations_csv}")

    # Coordinate extraction: check for separate lat/lon or packed spatial column
    if "latitude" in df_cameras.columns and "longitude" in df_cameras.columns:
        df_cameras["lat"] = df_cameras["latitude"].astype(float)
        df_cameras["lon"] = df_cameras["longitude"].astype(float)
    else:
        spatial_col = next(
            (c for c in ["location", "geo_point_2d", "geo_shape", "coordinates"] if c in df_cameras.columns),
            None,
        )
        if not spatial_col:
            raise KeyError(
                f"Could not find geographic coordinates. Available columns: {df_cameras.columns.tolist()}"
            )
        parsed_coords = df_cameras[spatial_col].apply(parse_geo_coordinates)
        df_cameras["lat"] = [p[0] for p in parsed_coords]
        df_cameras["lon"] = [p[1] for p in parsed_coords]

    # Ensure required placement description exists
    if "sensor_placement" not in df_cameras.columns:
        df_cameras["sensor_placement"] = df_cameras["camera_id"]

    # -------------------------------------------------------------
    # 2. Process Raw Crowd Telemetry (43-column dataset)
    # -------------------------------------------------------------
    df_raw = load_csv_auto_delim(telemetry_csv)
    df_raw.columns = df_raw.columns.str.strip()

    # Locate timestamp column
    time_col = next(
        (c for c in ["createdAt", "created_at", "timestamp", "time"] if c in df_raw.columns),
        df_raw.columns[0],
    )

    # Filter for pedestrian counting tripwires (line0, line1, line2)
    pedestrian_cols = [c for c in df_raw.columns if re.search(r"camera\d+_line\d+_(in|out)", c, re.IGNORECASE)]

    # Fallback to all directional streams if line tripwires are not explicitly labeled
    if not pedestrian_cols:
        pedestrian_cols = [
            c for c in df_raw.columns if c != time_col and (c.endswith("_in") or c.endswith("_out"))
        ]

    # Unpivot wide columns to relational records
    df_unpivoted = df_raw.melt(
        id_vars=[time_col],
        value_vars=pedestrian_cols,
        var_name="measurement_stream",
        value_name="pedestrian_count",
    )

    # Extract camera identifier (e.g., 'camera0', 'camera1')
    df_unpivoted["camera_id"] = (
        df_unpivoted["measurement_stream"]
        .str.extract(r"(camera\d+|sensor\d+)", flags=re.IGNORECASE)[0]
        .str.lower()
    )

    # Classify directional flow
    df_unpivoted["direction"] = df_unpivoted["measurement_stream"].apply(
        lambda x: "current_inflow" if str(x).endswith("_in") else "current_outflow"
    )

    # Aggregate total inflow and outflow per timestamp per node
    df_agg = (
        df_unpivoted.pivot_table(
            index=[time_col, "camera_id"],
            columns="direction",
            values="pedestrian_count",
            aggfunc="sum",
        )
        .reset_index()
    )

    # Normalize temporal values with UTC to handle daylight saving offsets
    df_agg = df_agg.rename(columns={time_col: "created_at"})
    df_agg["created_at_dt"] = pd.to_datetime(df_agg["created_at"], utc=True)
    df_agg["current_inflow"] = df_agg.get("current_inflow", 0.0).fillna(0.0)
    df_agg["current_outflow"] = df_agg.get("current_outflow", 0.0).fillna(0.0)

    # Order chronologically descending
    df_agg = df_agg.sort_values(by=["camera_id", "created_at_dt"], ascending=[True, False]).reset_index(drop=True)

    # -------------------------------------------------------------
    # 3. Construct Primary Keys & Risk Thresholds
    # -------------------------------------------------------------
    df_agg["chokepoint_pk"] = (
        df_agg["camera_id"] + "_" + df_agg["created_at_dt"].dt.strftime("%Y-%m-%dT%H:%M:%S.000Z")
    )

    net_flux = df_agg["current_inflow"] - df_agg["current_outflow"]
    df_agg["risk_level"] = net_flux.apply(
        lambda q: "CRITICAL" if q > 40 else ("ELEVATED" if q > 15 else "NOMINAL")
    )
    df_agg["safety_directive"] = ""

    # Clean ISO date format for SQLite storage
    df_agg["created_at"] = df_agg["created_at_dt"].dt.strftime("%Y-%m-%d %H:%M:%S")

    # -------------------------------------------------------------
    # 4. Database Persistence & Index Creation
    # -------------------------------------------------------------
    os.makedirs(os.path.dirname(os.path.abspath(db_path)), exist_ok=True)
    conn = sqlite3.connect(db_path)

    # Table 1: Sensor Nodes (Parent Metadata)
    df_cameras[["camera_id", "sensor_placement", "lat", "lon"]].to_sql(
        "sensor_nodes", conn, if_exists="replace", index=False
    )

    # Table 2: Crowd Chokepoints (Child Telemetry Logs)
    telemetry_cols = [
        "chokepoint_pk",
        "camera_id",
        "created_at",
        "current_inflow",
        "current_outflow",
        "risk_level",
        "safety_directive",
    ]
    df_agg[telemetry_cols].to_sql("crowd_chokepoints", conn, if_exists="replace", index=False)

    # Index for high-speed chronological queries
    cursor = conn.cursor()
    cursor.execute(
        "CREATE INDEX IF NOT EXISTS idx_camera_created_desc ON crowd_chokepoints(camera_id, created_at DESC);"
    )
    conn.commit()
    conn.close()

    print(
        f"ETL Complete: Saved {len(df_cameras)} sensors and {len(df_agg)} chokepoint logs into '{db_path}'."
    )


if __name__ == "__main__":
    def resolve_path(filename: str) -> str:
        candidates = [
            os.path.join("data", filename),
            filename,
            os.path.join("..", "data", filename),
        ]
        return next((p for p in candidates if os.path.exists(p)), os.path.join("data", filename))

    # Handles your specific Ghent dataset filenames
    telemetry_filename = "druktemeting-gent-kortenmarkt-2022-februari-maart.csv"
    if not os.path.exists(resolve_path(telemetry_filename)):
        telemetry_filename = "druktemeting-gent-kortedagsteeg.csv"

    run_etl(
        telemetry_csv=resolve_path(telemetry_filename),
        camera_locations_csv=resolve_path("camera_locations.csv"),
        db_path=os.path.join("data", "ghent_telemetry.db"),
    )
    