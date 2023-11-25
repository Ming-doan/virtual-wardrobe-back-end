# Get all item APIs

from firebase_option import db


def get_all(collection_name='Products'):
    collection = db.collection(collection_name)
    # Get all docs
    documents = collection.stream()
    docs = []
    for doc in documents:
        dictionary = doc.to_dict()
        dictionary['id'] = doc.id
        docs.append(dictionary)
    return docs


def get_by_id(id: str, collection_name='Products'):
    collection = db.collection(collection_name)
    document = collection.document(id).get()
    if document.exists:
        data = document.to_dict()
        data['id'] = id
        return data
    else:
        raise Exception(f'Document {id} not found!')
