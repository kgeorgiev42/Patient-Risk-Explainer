from flask import Flask
from src.config.config import config_dict

def create_app(config_name):
    app = Flask(__name__)
    app.config.from_object(config_dict[config_name])
    config_dict[config_name].init_app(app)

    ### initialize blueprints
    from src.main import bp as views_bp
    app.register_blueprint(views_bp)

    return app


import src.main
