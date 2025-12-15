
import sqlite3
import os

db_paths = {
    "head_hunter": "database/head_hunter_jobs.db",
    "job_post_api": "database/job_post_api.db",
    "job_seeker": "database/job_seeker.db"
}

results = {}

for name, path in db_paths.items():
    full_path = os.path.join("/workspace", path)  # Force absolute path
    if os.path.exists(full_path):
        try:
            conn = sqlite3.connect(full_path)
            c = conn.cursor()
            
            # List tables
            c.execute("SELECT name FROM sqlite_master WHERE type='table';")
            tables = c.fetchall()
            results[name] = {"tables": tables, "counts": {}}
            
            for table in tables:
                table_name = table[0]
                try:
                    c.execute(f"SELECT COUNT(*) FROM {table_name}")
                    count = c.fetchone()[0]
                    results[name]["counts"][table_name] = count
                except:
                    results[name]["counts"][table_name] = "Error"
            
            conn.close()
        except Exception as e:
            results[name] = f"Error connecting: {e}"
    else:
        results[name] = "File not found"

print(results)
