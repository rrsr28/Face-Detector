from flask import Flask, render_template, request
import cv2
import os
from PIL import Image

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/detect', methods=['POST'])
def detect():
    # Get the uploaded image file
    image_file = request.files['image']

    # Save the image file to a temporary location
    image_path = 'temp.jpg'
    image_file.save(image_path)

    # Load the pre-trained face detection model
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    # Read the image
    image = cv2.imread(image_path)

    # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Detect faces in the image
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    # Draw bounding boxes around the detected faces
    for (x, y, w, h) in faces:
        cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)

    # Save the output image with the bounding boxes
    output_filename = 'output.jpg'
    output_path = os.path.join(app.static_folder, output_filename)
    cv2.imwrite(output_path, image)

    # Remove the temporary image file
    os.remove(image_path)

    return render_template('result.html', image_path=output_path)

if __name__ == '__main__':
    app.run(debug=True)