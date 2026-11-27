"""When someone last used a project, for the nightly crawl reminders (scripts/nightly_crawl.py).

Any successful request of a logged-in browser that names a project counts: opening the project page, a card
action, a note, a chat question. It resets the reminder count, so the next reminder is 30 days away again.
"""
from datetime import datetime, timedelta

from flask import request, session

from app import db

WRITE_EVERY = timedelta(hours=1)  # one DB write per project and hour at most; reading pages polls a lot

REMIND_DAYS = (30, 60)  # "do you still need the nightly crawl?" emails
OFF_DAYS = 90           # the nightly crawl turns off by itself


def _project_id() -> int | None:
    args = request.view_args or {}
    if "slug" in args:
        row = db.session.execute(db.text("SELECT id FROM projects WHERE slug = :s"), {"s": args["slug"]}).first()
        return row[0] if row else None
    if "pp_id" in args:
        row = db.session.execute(db.text("SELECT project_id FROM project_papers WHERE id = :i"),
                                 {"i": args["pp_id"]}).first()
        return row[0] if row else None
    if "msg_id" in args:
        row = db.session.execute(db.text(
            "SELECT pp.project_id FROM chat_messages m JOIN project_papers pp ON pp.id = m.project_paper_id "
            "WHERE m.id = :i"), {"i": args["msg_id"]}).first()
        return row[0] if row else None
    return request.values.get("project_id", type=int)


def record(response):
    """after_request hook: note the activity on the request's project."""
    if not (200 <= response.status_code < 300) or "logged_in" not in session:
        return response
    project_id = _project_id()
    if project_id is None:
        return response
    now = datetime.utcnow()
    row = db.session.execute(db.text(
        "SELECT last_activity_at, inactivity_reminders FROM projects WHERE id = :i"), {"i": project_id}).first()
    if row is None:
        return response
    last = datetime.fromisoformat(str(row[0])) if row[0] else None
    if last is None or now - last >= WRITE_EVERY or row[1]:
        # Own connection: the route's session is done, and nothing of it should be committed here
        with db.engine.begin() as conn:
            conn.execute(db.text("UPDATE projects SET last_activity_at = :now, inactivity_reminders = 0 "
                                 "WHERE id = :i"), {"now": now, "i": project_id})
    return response


def days_idle(project, now: datetime | None = None) -> int:
    since = project.last_activity_at or project.created_at or datetime.utcnow()
    return ((now or datetime.utcnow()) - since).days
