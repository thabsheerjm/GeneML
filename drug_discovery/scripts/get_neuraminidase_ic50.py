import sqlite3
import pandas as pd
import os

db_path = "data/db/chembl_36.db"
sql_file = "sql/influenza_neuraminidase_ic50.sql"
csv_out = "data/raw/influenza_neuraminidase_ic50.csv"

if __name__ == '__main__':
    os.makedirs("data/raw", exist_ok=True)
    with open(sql_file) as f:
        query = f.read()

    connect = sqlite3.connect(db_path)
    df = pd.read_sql(query, connect)
    connect.close()

    df.to_csv(csv_out, index=False)
    print(f"Saved {len(df)} rows to {csv_out}")
