# Reconization routes

from flask import Blueprint, request
from logic.emotion_detection import predict_emotion, get_emotion_mark
from logic.collaborarive_filtering import add_to_matrix
from utils.response_object import response_object
from utils.convert import base64_to_numpy


recognization = Blueprint('recognize', __name__)


@recognization.route('/', methods=['POST'])
def recognize():
    # Get request data
    request_data = request.get_json()
    image = base64_to_numpy(request_data['image'])
    cloth_id = request_data['cloth_id']

    # Analyze request data
    result = predict_emotion(image)
    emotion = list(result)[0]['emotion']

    # Get result addition mark
    mark = get_emotion_mark(emotion)
    add_to_matrix(cloth_id, mark=mark)

    return response_object(message="Fucking success", data=emotion)
