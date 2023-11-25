# Get list of items

from flask import Blueprint
from firebase_option import db
from utils.response_object import response_object


getitem = Blueprint('items', __name__)
# Define clothes collection
clothes = db.collection('Products')


@getitem.route('/')
def get_items():
    # Fetch data from database
    docs = clothes.stream()
    documents = [doc.to_dict() for doc in docs]

    # Sort data as per user's preference

    # Return data
    return response_object(message="Fucking success", data=documents)


@getitem.route('/<string:item_id>')
def get_item(item_id):
    # Fetch data from database
    doc_snapshot = clothes.document(item_id).get()
    if doc_snapshot.exists:
        doc_data = doc_snapshot.to_dict()
        return response_object(message="Fucking success", data=doc_data)
    else:
        return response_object(message=f"Not found document {item_id}")
