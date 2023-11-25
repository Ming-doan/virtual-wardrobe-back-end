# Back-end APIs for Virtual Wardrobe

from flask import Flask
from routes import api

# Initialize Flask app
app = Flask(__name__)


# Default route
@app.route('/')
def hello_world():
    return 'Back-end APIs for Virtual Wardrobe'


# Adding blueprints
app.register_blueprint(api, url_prefix='/api')
