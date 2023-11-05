from flask import Flask, render_template, request, jsonify
import cv2
import base64
import numpy as np

app = Flask(__name__)

def detect_faces(image):

    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    for (x, y, w, h) in faces:
        cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)

    return image

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/detect', methods=['POST'])
def detect():

    image_data = request.form['imageData']
    image_data = base64.b64decode(image_data)

    image_array = np.frombuffer(image_data, np.uint8)

    image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
    result = detect_faces(image)

    # Encode the result image as base64
    _, result_data = cv2.imencode('.jpg', result)
    result_base64 = base64.b64encode(result_data).decode('utf-8')

    return jsonify(result=result_base64)

if __name__ == '__main__':
    app.run(debug=True)
