from flask import Flask
import os
from flask.ext.cors import CORS

FLASK_NAME = "MIT_AI_FB2019"
FLASK_SECRET_KEY = "fluffybunny"


def create_app():
    app = Flask(FLASK_NAME)
    app.secret_key = FLASK_SECRET_KEY
    return app


app = create_app()
cors = CORS(app)