from .facial_analysis import FacialImageProcessing
import os
import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.compat.v1.keras.backend import set_session

config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.compat.v1.Session(config=config)
set_session(sess)

imgProcessing = FacialImageProcessing(False)

INPUT_SIZE = (224, 224)

model_name = 'mobilenet_7.h5'  # 'mobilenet_7.h5'
models_path, _ = os.path.split(os.path.realpath(__file__))
models_path = os.path.join(models_path, '..', 'model')
model_file = os.path.join(models_path, model_name)

model = load_model(model_file)

idx_to_class = {0: 'Anger', 1: 'Disgust', 2: 'Fear',
                3: 'Happiness', 4: 'Neutral', 5: 'Sadness', 6: 'Surprise'}


def predict_emotion(img, idx_to_class=idx_to_class):
    # frame=cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
    bounding_boxes, points = imgProcessing.detect_faces(img)
    if len(bounding_boxes) == 0:
        return None
    points = points.T
    for bbox, p in zip(bounding_boxes, points):
        box = bbox.astype(np.int32)
        x1, y1, x2, y2 = box[0:4]
        face_img = img[y1:y2, x1:x2, :]

        try:
            face_img = cv2.resize(face_img, INPUT_SIZE)
        except:
            continue
        inp = face_img.astype(np.float32)
        inp[..., 0] -= 103.939
        inp[..., 1] -= 116.779
        inp[..., 2] -= 123.68
        inp = np.expand_dims(inp, axis=0)
        scores = model.predict(inp)[0]
        idx = np.argmax(scores)
        emotion = idx_to_class[idx]

        obj = {}
        obj['emotion'] = emotion
        obj['score'] = scores[idx]
        obj['bbox'] = box[0:4]

        yield obj


def get_emotion_mark(emotion):
    if emotion == 'Happiness':
        return 2
    elif emotion == 'Surprise':
        return 1
    elif emotion == 'Sadness':
        return -1
    elif emotion == 'Anger':
        return -2
    else:
        return 0
