from flask import Blueprint

projects_bp = Blueprint("projects", __name__, url_prefix="/projects")
papers_bp = Blueprint("papers", __name__)
crawl_bp = Blueprint("crawl", __name__)

from app.routes import projects, papers, crawl