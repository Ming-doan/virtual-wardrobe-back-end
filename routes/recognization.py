# Reconization routes

from flask import Blueprint, request


recognization = Blueprint('recognize', __name__)


@recognization.route('/', methods=['POST'])
def recognize():
    # Get request data

    # Analyze request data

    return
