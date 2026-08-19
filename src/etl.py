import pandas as pd
import re
import sqlite3

def run_pipeline(raw_csv_path: str, db_path: str):
    df = pd.read_csv(raw_csv_path)
    
    # 1. Regex parse WKT string coordinates into numeric lat/lon
    df['lat'] = df['location'].apply(lambda x: float(re.findall(r"[-+]?\d*\.\d+|\d+", str(x))[0]))
    df['lon'] = df['location'].apply(lambda x: float(re.findall(r"[-+]?\d*\.\d+|\d+", str(x))[1]))
    
    # 2. Build composite primary key
    df['chokepoint_pk'] = df['camera_id'] + "_" + df['createdat'].astype(str)
    
    # 3. Export to SQLite database
    conn = sqlite3.connect(db_path)
    df.to_sql("crowd_chokepoints", conn, if_exists="replace", index=False)
    conn.close()