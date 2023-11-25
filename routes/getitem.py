# Get list of items

from flask import Blueprint


getitem = Blueprint('items', __name__)


@getitem.route('/')
def get_items():
    # Fetch data from database

    # Sort data as per user's preference

    # Return data
    return 'Get list of items'


@getitem.route('/<string:item_id>')
def get_item(item_id):
    # Fetch data from database

    # Return data
    return 'Get item with id {}'.format(item_id)
