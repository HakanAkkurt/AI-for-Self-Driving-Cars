import numpy as np
import socketio
import eventlet

from flask import Flask
from flask_meld import Meld
from keras.models import load_model
import base64
from io import BytesIO
from PIL import Image
import cv2
import os

# pip install flask
# pip install python-socketio
# pip install eventlet
# pip install Meld
# pip install flask_meld
# Can't connect fix: 
# pip install python-engineio==3.13.2
# pip install python-socketio==4.6.1

sio = socketio.Server()

app = Flask(__name__) # '__main__'
app.config['SECRET_KEY'] = 'big!secret'

meld = Meld()
meld.init_app(app)

speed_limit = 20

def img_preprocess(img):
  img = img[60:135,:,:]
  img = cv2.cvtColor(img, cv2.COLOR_RGB2YUV)
  img = cv2.GaussianBlur(img, (3, 3), 0)
  img = cv2.resize(img, (200, 66))
  img = img/255
  return img

# Communicating between the model and the simulator
@sio.on('telemetry')
def telemetry(sid, data):
    speed = float(data['speed'])
    image = Image.open(BytesIO(base64.b64decode(data['image'])))
    image = np.asarray(image)
    image = img_preprocess(image)
    image = np.array([image])
    steering_angle = float(model.predict(image))
    throttle = 1.0 - speed/speed_limit
    print('{} {} {}'.format(steering_angle, throttle, speed))
    send_control(steering_angle, throttle)

@sio.on('connect')
def connect(sid, environ):
    print('Connected', sid)
    send_control(0, 0)

def send_control(steering_angle, throttle):
    sio.emit('steer', data = {
        'steering_angle': steering_angle.__str__(),
        'throttle': throttle.__str__()
    })

if __name__ == '__main__':
    THIS_FOLDER = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(THIS_FOLDER, 'model.h5')
    model = load_model(model_path)
    app.wsgi_app = socketio.WSGIApp(sio, app.wsgi_app)
    eventlet.wsgi.server(eventlet.listen(('', 4567)), app)
