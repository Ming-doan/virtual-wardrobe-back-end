# Response object for API

from flask import jsonify


def response_object(status=200, message="", data=None):
    return jsonify({
        'status': status,
        'message': message,
        'data': data
    })
