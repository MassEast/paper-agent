from flask import render_template

from app.auth import login_required
from app.models import CrawlLog
from app.routes import crawl_bp, projects_bp
from app.crawl import _crawl_current_papers, _crawl_in_flight


@crawl_bp.route("/project/<slug>/crawl-status")
@login_required
def status(slug):
    from app.models import Project
    project = Project.query.filter_by(slug=slug).first_or_404()

    crawl = CrawlLog.query.filter_by(
        project_id=project.id
    ).order_by(
        CrawlLog.started_at.desc()
    ).first()

    current_paper = _crawl_current_papers.get(crawl.id, "") if crawl else ""
    in_flight = _crawl_in_flight.get(crawl.id, 0) if crawl else 0
    return render_template(
        "partials/crawl_status.html",
        project=project,
        crawl=crawl,
        current_paper=current_paper,
        in_flight=in_flight,
    )


@projects_bp.route("/<slug>/crawl-status")
@login_required
def status_projects(slug):
    from app.models import Project
    project = Project.query.filter_by(slug=slug).first_or_404()

    crawl = CrawlLog.query.filter_by(
        project_id=project.id
    ).order_by(
        CrawlLog.started_at.desc()
    ).first()

    current_paper = _crawl_current_papers.get(crawl.id, "") if crawl else ""
    in_flight = _crawl_in_flight.get(crawl.id, 0) if crawl else 0
    return render_template(
        "partials/crawl_status.html",
        project=project,
        crawl=crawl,
        current_paper=current_paper,
        in_flight=in_flight,
    )
