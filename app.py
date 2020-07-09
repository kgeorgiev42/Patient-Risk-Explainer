# application.py
import os
from os import environ

from src import create_app

HOST = environ.get('SERVER_HOST', 'localhost')
try:
    PORT = int(environ.get('SERVER_PORT', '5000'))
except ValueError:
    PORT = 5000

app = create_app(os.getenv('FLASK_ENV') or 'default')
app.run() #run app in debug mode on port
