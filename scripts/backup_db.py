#!/usr/bin/env python3
"""
Daily SQLite backup for the CronJob.

Snapshots the live DB via SQLite's online backup API (sqlite3.Connection.backup),
which is safe to run against a database that's actively being read from and
written to under WAL mode -- a plain file copy is not: WAL mode keeps recent
commits in a separate `-wal` file until checkpointed, so `cp related_work.db`
alone can silently miss recent writes or catch the file mid-checkpoint.

Writes to a separate PVC (not the one the live DB lives on) so a lost/corrupted
main volume doesn't take the backups down with it. Runs on its own daily
CronJob schedule -- not from within nightly_crawl.py, which fires up to 4x/day
(once per possible project crawl_hour) and would need its own once-per-day
guard to avoid duplicate backups; a dedicated job sidesteps that.

Run: python scripts/backup_db.py
Env: DATABASE_URL (source), BACKUP_DIR (default /backups),
     BACKUP_RETENTION_DAYS (default 30)
"""
import logging
import os
import sqlite3
import sys
from datetime import datetime, timezone
from logging.handlers import RotatingFileHandler

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

_log = logging.getLogger(__name__)


def _setup_logging():
    logs_dir = os.environ.get("LOGS_DIR", "/data/logs")
    os.makedirs(logs_dir, exist_ok=True)
    handler = RotatingFileHandler(f"{logs_dir}/backup.log", maxBytes=10 * 1024 * 1024, backupCount=5)
    handler.setLevel(logging.INFO)
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s: %(message)s"))
    root = logging.getLogger()
    root.setLevel(logging.INFO)
    root.addHandler(handler)
    root.addHandler(logging.StreamHandler(sys.stdout))


def _db_path() -> str:
    url = os.environ.get("DATABASE_URL", "sqlite:////tmp/test.db")
    if not url.startswith("sqlite:///"):
        raise EnvironmentError(f"backup_db.py only supports a sqlite DATABASE_URL, got: {url!r}")
    return url[len("sqlite:///"):]


def backup_once(db_path: str, backup_dir: str) -> str:
    """Write today's snapshot (no-op if it already exists) and verify it's readable."""
    os.makedirs(backup_dir, exist_ok=True)
    today = datetime.now(timezone.utc).strftime("%Y%m%d")
    dest_path = os.path.join(backup_dir, f"related_work_{today}.db")

    if os.path.exists(dest_path):
        _log.info("Backup for %s already exists at %s -- skipping", today, dest_path)
        return dest_path

    # mode=ro: the backup process should never be able to write to the live DB, even by accident.
    src = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    dst = sqlite3.connect(dest_path)
    try:
        src.backup(dst)
    finally:
        dst.close()
        src.close()

    # A corrupt backup sitting unnoticed for weeks defeats the whole point -- verify it now,
    # not at restore time.
    check_conn = sqlite3.connect(dest_path)
    try:
        result = check_conn.execute("PRAGMA integrity_check").fetchone()[0]
        if result != "ok":
            check_conn.close()
            os.remove(dest_path)
            raise RuntimeError(f"Backup integrity check failed for {dest_path}: {result}")
        paper_count = check_conn.execute("SELECT COUNT(*) FROM papers").fetchone()[0]
    finally:
        check_conn.close()

    size_mb = os.path.getsize(dest_path) / (1024 * 1024)
    _log.info("Backup written: %s (%.1f MB, %d papers, integrity ok)", dest_path, size_mb, paper_count)
    return dest_path


def prune_old_backups(backup_dir: str, retention_days: int):
    now = datetime.now(timezone.utc)
    removed = 0
    for fname in sorted(os.listdir(backup_dir)):
        if not (fname.startswith("related_work_") and fname.endswith(".db")):
            continue
        date_str = fname[len("related_work_"):-len(".db")]
        try:
            file_date = datetime.strptime(date_str, "%Y%m%d").replace(tzinfo=timezone.utc)
        except ValueError:
            continue  # not one of our dated snapshots -- leave it alone
        age_days = (now - file_date).days
        if age_days > retention_days:
            os.remove(os.path.join(backup_dir, fname))
            removed += 1
            _log.info("Pruned backup older than %d days: %s (age %d days)", retention_days, fname, age_days)
    _log.info("Pruned %d old backup(s), retention=%d days", removed, retention_days)


def main():
    _setup_logging()
    db_path = _db_path()
    backup_dir = os.environ.get("BACKUP_DIR", "/backups")
    retention_days = int(os.environ.get("BACKUP_RETENTION_DAYS", "30"))

    if not os.path.exists(db_path):
        _log.error("Source DB not found at %s -- nothing to back up", db_path)
        sys.exit(1)

    backup_once(db_path, backup_dir)
    prune_old_backups(backup_dir, retention_days)
    _log.info("Backup run complete.")


if __name__ == "__main__":
    main()
