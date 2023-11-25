# Content-based filtering

import numpy as np
from utils.api_helper import get_all, get_by_id
from utils.server_loader import write_json, write_numpy, read_numpy, load_json


def build_matrix():
    # Fetch all documents
    documents = get_all()
    cats_dictionary = {}
    items_dictionary = {}

    # M (length of document)
    M = len(documents)

    for doc in documents:
        if doc['id'] not in items_dictionary:
            items_dictionary[doc['id']] = len(items_dictionary)
        for cat in doc['categories']:
            if cat not in cats_dictionary:
                cats_dictionary[cat] = len(cats_dictionary)

    # N (length of categories)
    N = len(cats_dictionary)

    # Build matrix
    matrix = np.zeros((M, N))

    # Save matrix, cats_dictionary, items_dictionary
    write_numpy('matrix.npy', matrix)
    write_json('cats_dictionary.json', cats_dictionary)
    write_json('items_dictionary.json', items_dictionary)


def add_to_matrix(item_id, data=None, mark=1):
    if data is None:
        data = get_by_id(item_id)

    # Read matrix, cats, items
    matrix = read_numpy('matrix.npy')
    cats_dictionary = load_json('cats_dictionary.json')
    items_dictionary = load_json('items_dictionary.json')

    item_idx = items_dictionary[item_id]

    # Add data to matrix
    for cat in data['categories']:
        cat_idx = cats_dictionary[cat]
        matrix[item_idx, cat_idx] += mark

    # Save matrix
    write_numpy('matrix.npy', matrix)


def get_recommendation(documents):
    # Read matrix, items
    matrix = read_numpy('matrix.npy')

    # Calculate sum of preference categories
    preffer_vector = np.sum(matrix, axis=0, keepdims=True)

    # Calculate score for each item
    vector = matrix @ preffer_vector.T

    # Sort items by score
    indexes = np.argsort(vector.T.squeeze())[::-1]

    # Sort documents
    for index in indexes:
        yield documents[index]
