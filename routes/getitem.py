# Get list of items

from flask import Blueprint
from firebase_option import db
from utils.response_object import response_object
from utils.api_helper import get_all, get_by_id
from logic.collaborarive_filtering import get_recommendation, add_to_matrix, build_matrix


getitem = Blueprint('items', __name__)
# Build matrix
build_matrix()


@getitem.route('/')
def get_items():
    # Fetch data from database
    documents = get_all()

    # Sort data as per user's preference
    documents = list(get_recommendation(documents))

    # Return data
    return response_object(message="Fucking success", data=documents)


@getitem.route('/<string:item_id>')
def get_item(item_id):
    # Fetch data from database
    data = get_by_id(item_id)
    add_to_matrix(item_id, data)
    return response_object(message="Fucking success", data=data)
