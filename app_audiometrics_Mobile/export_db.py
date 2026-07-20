"""Export the Audio Metrics SQLite database to JSON.

Usage:
    python export_db.py                          # audio_metrics.db -> sessions.json
    python export_db.py mydata.db out.json       # explicit input/output
    python export_db.py --table sessions         # choose a table
    python export_db.py --stdout                 # print JSON to stdout instead
"""

import argparse
import json
import sqlite3
import sys
from pathlib import Path

APP_DIR = Path(__file__).parent
DEFAULT_DB = APP_DIR / "audio_metrics.db"
DEFAULT_OUT = APP_DIR / "sessions.json"


def export_table(db_path, table):
    """Return a list of row dicts for `table`, ordered by rowid."""
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row  # rows behave like dicts (column-name keys)
    try:
        # Validate the table exists (and guard against SQL injection on the name)
        names = {r[0] for r in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        )}
        if table not in names:
            raise SystemExit(
                f"Table '{table}' not found in {db_path}. "
                f"Available: {', '.join(sorted(names)) or '(none)'}"
            )
        rows = conn.execute(f"SELECT * FROM {table}").fetchall()
        return [dict(r) for r in rows]
    finally:
        conn.close()


def main():
    parser = argparse.ArgumentParser(description="Export Audio Metrics SQLite DB to JSON.")
    parser.add_argument("db", nargs="?", default=str(DEFAULT_DB),
                        help=f"SQLite file (default: {DEFAULT_DB.name})")
    parser.add_argument("out", nargs="?", default=str(DEFAULT_OUT),
                        help=f"Output JSON file (default: {DEFAULT_OUT.name})")
    parser.add_argument("--table", default="sessions", help="Table to export (default: sessions)")
    parser.add_argument("--stdout", action="store_true", help="Write JSON to stdout instead of a file")
    args = parser.parse_args()

    db_path = Path(args.db)
    if not db_path.exists():
        raise SystemExit(f"Database not found: {db_path}")

    records = export_table(db_path, args.table)
    payload = json.dumps(records, indent=2, ensure_ascii=False)

    if args.stdout:
        sys.stdout.write(payload + "\n")
    else:
        out_path = Path(args.out)
        out_path.write_text(payload, encoding="utf-8")
        print(f"Exported {len(records)} row(s) from '{args.table}' -> {out_path}")


if __name__ == "__main__":
    main()
