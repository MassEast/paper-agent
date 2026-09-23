import os
import logging
from logging.handlers import RotatingFileHandler
from datetime import date, timedelta, timezone
from zoneinfo import ZoneInfo

from flask import Flask
from flask_sqlalchemy import SQLAlchemy
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address

db = SQLAlchemy()
limiter = Limiter(key_func=get_remote_address, default_limits=[])

LOGS_DIR = os.environ.get(
    "LOGS_DIR",
    os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "logs"),
)


def setup_logging(app: Flask):
    os.makedirs(LOGS_DIR, exist_ok=True)
    
    file_handler = RotatingFileHandler(
        f"{LOGS_DIR}/app.log",
        maxBytes=10 * 1024 * 1024,
        backupCount=5
    )
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(logging.Formatter(
        "%(asctime)s %(levelname)s: %(message)s [in %(pathname)s:%(lineno)d]"
    ))
    
    root_logger = logging.getLogger()
    # Only add once — root logger covers all app.* child loggers via propagation
    if not any(isinstance(h, RotatingFileHandler) for h in root_logger.handlers):
        root_logger.addHandler(file_handler)
    if root_logger.level == logging.NOTSET or root_logger.level > logging.INFO:
        root_logger.setLevel(logging.INFO)

    # Werkzeug logs every HTTP request at INFO — too noisy for app.log
    logging.getLogger('werkzeug').setLevel(logging.WARNING)

    # Do NOT add a second handler to app.logger — it propagates to root and would duplicate every line
    app.logger.setLevel(logging.INFO)
    app.logger.info("Related Work Agent startup")
    
    return file_handler


def create_app():
    app = Flask(__name__)

    SECRET_KEY = os.environ.get("SECRET_KEY")
    SITE_PASSWORD = os.environ.get("SITE_PASSWORD")
    if not SECRET_KEY:
        raise EnvironmentError("SECRET_KEY environment variable is required")
    if not SITE_PASSWORD:
        raise EnvironmentError("SITE_PASSWORD environment variable is required")
    app.config["SECRET_KEY"] = SECRET_KEY
    app.config["SITE_PASSWORD"] = SITE_PASSWORD
    app.config["SQLALCHEMY_DATABASE_URI"] = os.environ.get(
        "DATABASE_URL", "sqlite:////tmp/test.db"
    )
    app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False

    app.config["SESSION_TYPE"] = "filesystem"
    app.config["SESSION_PERMANENT"] = True
    app.config["PERMANENT_SESSION_LIFETIME"] = timedelta(days=30)

    db.init_app(app)
    limiter.init_app(app)

    from app.models import Project, Paper, PaperSummary, ProjectPaper, CrawlLog
    from sqlalchemy import event as _sa_event

    with app.app_context():
        # Set SQLite PRAGMAs on every new connection — busy_timeout is per-connection,
        # so setting it once at startup only covered the first connection in the pool.
        @_sa_event.listens_for(db.engine, "connect")
        def _set_sqlite_pragmas(dbapi_conn, _record):
            cur = dbapi_conn.cursor()
            cur.execute("PRAGMA journal_mode=WAL")
            cur.execute("PRAGMA busy_timeout=30000")
            cur.close()

        db.create_all()

        # Schema migrations — safe to run repeatedly; errors on existing columns are swallowed
        _migrations = [
            "ALTER TABLE projects ADD COLUMN saved_keywords TEXT",
            "ALTER TABLE projects ADD COLUMN crawl_hour INTEGER",
            "ALTER TABLE paper_summaries ADD COLUMN tokens_used INTEGER DEFAULT 0",
            "ALTER TABLE crawl_logs ADD COLUMN paper_limit INTEGER DEFAULT 0",
            "ALTER TABLE crawl_logs ADD COLUMN sources_used TEXT",
            "ALTER TABLE papers ADD COLUMN page_count INTEGER",
            "ALTER TABLE projects ADD COLUMN trashed_at DATETIME",
            "ALTER TABLE papers ADD COLUMN figure_caption TEXT",
            "CREATE TABLE IF NOT EXISTS screened_papers (id INTEGER PRIMARY KEY AUTOINCREMENT, project_id INTEGER NOT NULL REFERENCES projects(id), arxiv_id TEXT NOT NULL, screening_hash TEXT NOT NULL, is_relevant INTEGER NOT NULL DEFAULT 0, screened_at DATETIME DEFAULT CURRENT_TIMESTAMP)",
            "CREATE INDEX IF NOT EXISTS idx_screened_lookup ON screened_papers(project_id, arxiv_id, screening_hash)",
            "ALTER TABLE papers ADD COLUMN citation_fetched_at DATETIME",
            "ALTER TABLE project_papers ADD COLUMN paper_tags TEXT DEFAULT '[]'",
            "ALTER TABLE papers ADD COLUMN enrich_complete INTEGER DEFAULT 0",
            "ALTER TABLE papers ADD COLUMN citation_manual BOOLEAN DEFAULT 0",
            "ALTER TABLE project_papers ADD COLUMN notes_height INTEGER",
            "ALTER TABLE project_papers ADD COLUMN collected_at DATETIME",
            # One-time backfill: added_at used to be overwritten on every move into My Collection,
            # so for already-curated papers it currently holds the collection date, not the crawl date.
            "UPDATE project_papers SET collected_at = added_at WHERE manual_tag IS NOT NULL AND collected_at IS NULL",
        ]
        for _sql in _migrations:
            try:
                db.session.execute(db.text(_sql))
                db.session.commit()
            except Exception:
                db.session.rollback()

        # Back-fill crawl_hour for existing projects that don't have one
        import random as _random
        try:
            unassigned = Project.query.filter(Project.crawl_hour == None).all()  # noqa: E711
            if unassigned:
                taken = {p.crawl_hour for p in Project.query.all() if p.crawl_hour is not None}
                for p in unassigned:
                    available = [h for h in range(1, 5) if h not in taken]
                    p.crawl_hour = _random.choice(available) if available else _random.randint(1, 4)
                    taken.add(p.crawl_hour)
                db.session.commit()
        except Exception:
            db.session.rollback()

        # Re-enrich papers with missing authors — can happen if the arXiv enrichment thread
        # hit a 429 at startup. Scholar (institutions/citations) runs separately so those
        # papers end up partially enriched. Only kick off threads for visibly broken papers
        # (missing authors); the nightly backfill handles the full enrich_complete sweep.
        _needs_author_enrichment: list[tuple[int, str, int]] = []
        try:
            from app.models import Paper, ProjectPaper
            _author_gaps = db.session.execute(db.text(
                "SELECT DISTINCT p.id, p.arxiv_id, pp.project_id "
                "FROM papers p JOIN project_papers pp ON pp.paper_id = p.id "
                "WHERE p.enrich_complete = 0 "
                "AND (p.authors IS NULL OR p.authors = '[]' OR p.authors = '') "
                "AND p.arxiv_id NOT LIKE 'web:%'"
            )).fetchall()
            for row in _author_gaps:
                _needs_author_enrichment.append((row[2], row[1], row[0]))
        except Exception:
            pass

    from app.auth import auth_bp
    from app.routes import projects_bp, papers_bp, crawl_bp

    app.register_blueprint(auth_bp)
    app.register_blueprint(projects_bp)
    app.register_blueprint(papers_bp)
    app.register_blueprint(crawl_bp)

    app.jinja_env.globals["date"] = date
    app.jinja_env.globals["timedelta"] = timedelta

    _berlin = ZoneInfo("Europe/Berlin")

    @app.template_filter("berlin_time")
    def berlin_time_filter(dt, fmt="%b %d, %H:%M"):
        if dt is None:
            return ""
        return dt.replace(tzinfo=timezone.utc).astimezone(_berlin).strftime(fmt)

    @app.route("/")
    def index():
        from flask import redirect
        return redirect("/projects")

    setup_logging(app)

    if _needs_author_enrichment:
        import threading as _threading
        import time as _time
        import logging as _log_e

        def _enrich_author_gaps(items, _app):
            _time.sleep(30)  # wait well past nightly crawl window before hitting arXiv
            _log_e.getLogger(__name__).info(
                "[startup] Re-enriching %d papers missing authors", len(items)
            )
            for project_id, arxiv_id, paper_id in items:
                try:
                    from app.crawl import start_collection_add_async
                    start_collection_add_async(project_id, arxiv_id, paper_id, _app)
                    _time.sleep(5)  # stagger arXiv requests
                except Exception:
                    pass

        _threading.Thread(
            target=_enrich_author_gaps,
            args=(_needs_author_enrichment, app),
            daemon=True,
        ).start()

    return app