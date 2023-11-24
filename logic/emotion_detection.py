import os
import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras.models import Model,Sequential, load_model,model_from_json

from tensorflow.compat.v1.keras.backend import set_session
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
sess=tf.compat.v1.Session(config=config)
set_session(sess)

from facial_analysis import FacialImageProcessing
imgProcessing=FacialImageProcessing(False)

INPUT_SIZE = (224, 224)

model_name='mobilenet_7.h5' # 'mobilenet_7.h5'
models_path,_ = os.path.split(os.path.realpath(__file__))
models_path=os.path.join(models_path,'..','model')
model_file=os.path.join(models_path,model_name)

model=load_model(model_file)

idx_to_class={0: 'Anger', 1: 'Disgust', 2: 'Fear', 3: 'Happiness', 4: 'Neutral', 5: 'Sadness', 6: 'Surprise'}

# using camera to capture video
cap = cv2.VideoCapture(0)
while True:
    ret, frame = cap.read()
    if not ret:
        break
    # frame=cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
    bounding_boxes, points = imgProcessing.detect_faces(frame)
    if len(bounding_boxes) == 0:
        continue
    points = points.T
    for bbox,p in zip(bounding_boxes, points):
        box=bbox.astype(np.int32)
        x1,y1,x2,y2=box[0:4]    
        face_img=frame[y1:y2,x1:x2,:]

        try:
            face_img=cv2.resize(face_img,INPUT_SIZE)
        except:
            continue
        inp=face_img.astype(np.float32)
        inp[..., 0] -= 103.939
        inp[..., 1] -= 116.779
        inp[..., 2] -= 123.68
        inp = np.expand_dims(inp, axis=0)
        scores=model.predict(inp)[0]
        idx=np.argmax(scores)
        emotion=idx_to_class[idx]
        cv2.rectangle(frame,(x1,y1),(x2,y2),(0,255,0),2)
        cv2.putText(frame,emotion,(x1,y1),cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),2)
    cv2.imshow('frame',frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
