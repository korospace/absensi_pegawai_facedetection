# Core
import os
from flask import Flask
from flask_restful import Api
from flask_sqlalchemy import SQLAlchemy
from flask_cors import CORS

# Route
from face_detection.routes import face_detection_routes

db = SQLAlchemy()

def create_app():
    # App initiation
    app = Flask(__name__, instance_relative_config=False)
    CORS(app)

    app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

    # Route initiation
    api = Api(app=app)
    face_detection_routes(api=api)

    # Database initiation
    db.init_app(app)

    with app.app_context():
        db.create_all()
        return app

if __name__ == "__main__":
    app = create_app()
    app.run(host="0.0.0.0", port=5000)
