import hmac
from functools import wraps

from flask import Blueprint, request, render_template, session, redirect, url_for, current_app
from app import limiter

auth_bp = Blueprint("auth", __name__, url_prefix="")


def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if "logged_in" not in session:
            return redirect(url_for("auth.login_page"))
        return f(*args, **kwargs)
    return decorated_function


@auth_bp.route("/login", methods=["GET"])
def login_page():
    return render_template("login.html")


@auth_bp.route("/login", methods=["POST"])
@limiter.limit("20 per hour; 5 per minute")
def login():
    password = request.form.get("password", "")
    if hmac.compare_digest(password, current_app.config["SITE_PASSWORD"]):
        session["logged_in"] = True
        return redirect(url_for("projects.index"))
    return render_template("login.html", error="Invalid password")


@auth_bp.route("/logout", methods=["POST"])
def logout():
    session.pop("logged_in", None)
    return redirect(url_for("auth.login_page"))