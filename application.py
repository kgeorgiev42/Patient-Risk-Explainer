# application.py
import os
from os import environ

from src import create_app

'''
HOST = environ.get('SERVER_HOST', 'localhost')
try:
    PORT = int(environ.get('SERVER_PORT', '5000'))
except ValueError:
    PORT = 5000
'''
application = create_app(os.getenv('FLASK_ENV') or 'default')
application.run(host='0.0.0.0', port=8080, debug=False)
