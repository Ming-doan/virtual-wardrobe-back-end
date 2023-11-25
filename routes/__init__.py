### API route initialization ###

from flask import Blueprint
from routes.getitem import getitem
from routes.recognization import recognization

api = Blueprint('api', __name__)

# Registering routes
api.register_blueprint(getitem, url_prefix='/items')
api.register_blueprint(recognization, url_prefix='/recognize')
