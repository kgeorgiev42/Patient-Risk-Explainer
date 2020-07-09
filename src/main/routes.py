import logging
from datetime import datetime
from logging.handlers import RotatingFileHandler

from flask import render_template

from src.main import bp

handler = RotatingFileHandler('application.log', maxBytes=10000, backupCount=3)
logger = logging.getLogger(__name__)
logger.setLevel(logging.ERROR)
logger.addHandler(handler)


@bp.route('/')
@bp.route('/home')
def home():
    """Renders the home page."""
    return render_template(
        'index.html',
        year=datetime.now().year,
    )
