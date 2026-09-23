"""
Tests for scripts/backup_db.py. Uses raw sqlite3 fixtures (not the Flask app/conftest
fixtures) since the backup script deliberately has no Flask/SQLAlchemy dependency --
it operates on the DB file directly, so tests should too.
"""
import os
import sqlite3
import sys
from datetime import datetime, timedelta, timezone

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.backup_db import _db_path, backup_once, prune_old_backups


@pytest.fixture
def source_db(tmp_path):
    """A minimal real sqlite file with a `papers` table (backup_once queries it for the log line)."""
    db_path = str(tmp_path / "related_work.db")
    conn = sqlite3.connect(db_path)
    conn.execute("CREATE TABLE papers (id INTEGER PRIMARY KEY, title TEXT)")
    conn.execute("INSERT INTO papers (title) VALUES ('Paper A'), ('Paper B')")
    conn.commit()
    conn.close()
    return db_path


class TestDbPath:
    def test_absolute_path_four_slashes(self, monkeypatch):
        monkeypatch.setenv("DATABASE_URL", "sqlite:////data/related_work.db")
        assert _db_path() == "/data/related_work.db"

    def test_relative_path_three_slashes(self, monkeypatch):
        monkeypatch.setenv("DATABASE_URL", "sqlite:///related_work.db")
        assert _db_path() == "related_work.db"

    def test_non_sqlite_url_rejected(self, monkeypatch):
        monkeypatch.setenv("DATABASE_URL", "postgresql://localhost/foo")
        with pytest.raises(EnvironmentError):
            _db_path()


class TestBackupOnce:
    def test_creates_snapshot_with_dated_filename(self, source_db, tmp_path):
        backup_dir = str(tmp_path / "backups")
        dest = backup_once(source_db, backup_dir)

        today = datetime.now(timezone.utc).strftime("%Y%m%d")
        assert dest == os.path.join(backup_dir, f"related_work_{today}.db")
        assert os.path.exists(dest)

        conn = sqlite3.connect(dest)
        rows = conn.execute("SELECT title FROM papers ORDER BY id").fetchall()
        conn.close()
        assert [r[0] for r in rows] == ["Paper A", "Paper B"]

    def test_second_call_same_day_is_noop(self, source_db, tmp_path):
        backup_dir = str(tmp_path / "backups")
        first = backup_once(source_db, backup_dir)
        mtime_before = os.path.getmtime(first)

        # Mutate the source after the first backup -- if the second call re-copied,
        # the backup would pick up "Paper C" too.
        conn = sqlite3.connect(source_db)
        conn.execute("INSERT INTO papers (title) VALUES ('Paper C')")
        conn.commit()
        conn.close()

        second = backup_once(source_db, backup_dir)
        assert second == first
        assert os.path.getmtime(second) == mtime_before

        conn = sqlite3.connect(second)
        count = conn.execute("SELECT COUNT(*) FROM papers").fetchone()[0]
        conn.close()
        assert count == 2  # still the pre-mutation snapshot -- not re-copied

    def test_source_not_writable_by_backup(self, source_db, tmp_path):
        """The source is opened mode=ro -- backup_once must never be able to write to it."""
        backup_dir = str(tmp_path / "backups")
        backup_once(source_db, backup_dir)

        conn = sqlite3.connect(source_db)
        count = conn.execute("SELECT COUNT(*) FROM papers").fetchone()[0]
        conn.close()
        assert count == 2  # untouched


class TestPruneOldBackups:
    def _make_backup_file(self, backup_dir, days_ago):
        date_str = (datetime.now(timezone.utc) - timedelta(days=days_ago)).strftime("%Y%m%d")
        path = os.path.join(backup_dir, f"related_work_{date_str}.db")
        sqlite3.connect(path).close()
        return path

    def test_removes_only_backups_older_than_retention(self, tmp_path):
        backup_dir = str(tmp_path / "backups")
        os.makedirs(backup_dir)
        recent = self._make_backup_file(backup_dir, days_ago=5)
        old = self._make_backup_file(backup_dir, days_ago=45)

        prune_old_backups(backup_dir, retention_days=30)

        assert os.path.exists(recent)
        assert not os.path.exists(old)

    def test_ignores_non_backup_files(self, tmp_path):
        backup_dir = str(tmp_path / "backups")
        os.makedirs(backup_dir)
        unrelated = os.path.join(backup_dir, "notes.txt")
        with open(unrelated, "w") as f:
            f.write("not a backup")

        prune_old_backups(backup_dir, retention_days=30)

        assert os.path.exists(unrelated)

    def test_boundary_exactly_at_retention_is_kept(self, tmp_path):
        backup_dir = str(tmp_path / "backups")
        os.makedirs(backup_dir)
        boundary = self._make_backup_file(backup_dir, days_ago=30)

        prune_old_backups(backup_dir, retention_days=30)

        assert os.path.exists(boundary)
