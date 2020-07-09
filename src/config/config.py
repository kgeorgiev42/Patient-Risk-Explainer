# log to stderr
import logging
import os
from logging import StreamHandler

from dotenv import load_dotenv

basedir = os.path.abspath(os.path.dirname(__file__))
load_dotenv(os.path.join(basedir, '.env'))


class Config(object):
    SECRET_KEY = os.urandom(16)

    # AWS_ACCESS_KEY_ID = os.environ.get('AWS_ACCESS_KEY_ID')
    # AWS_SECRET_ACCESS_KEY = os.environ.get('AWS_SECRET_ACCESS_KEY')
    # AWS_SESSION_TOKEN = os.environ.get('AWS_SESSION_TOKEN')

    @staticmethod
    def init_app(app):
        file_handler = StreamHandler()
        file_handler.setLevel(logging.INFO)
        app.logger.addHandler(file_handler)


class DevelopmentConfig(Config):
    DEBUG = True


class TestingConfig(Config):
    TESTING = True


class DeploymentConfig(Config):
    DEPLOY = True


config_dict = {
    'development': DevelopmentConfig,
    'testing': TestingConfig,
    'deployment': DeploymentConfig,
    'default': DevelopmentConfig
}
