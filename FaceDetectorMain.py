import cv2

def detect_faces(image):
    # Load the pre-trained face detection model
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    for (x, y, w, h) in faces:
        cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)

    return image

def detect_faces_in_image(image_path):

    image = cv2.imread(image_path)
    result = detect_faces(image)

    cv2.imshow('Face Detection', result)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def detect_faces_in_video(video_path):
    
    video = cv2.VideoCapture(video_path)

    while True:
        ret, frame = video.read()

        if not ret:
            break
        
        result = detect_faces(frame)
        cv2.imshow('Face Detection', result)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    video.release()
    cv2.destroyAllWindows()

image_path = 'Images/Crowd1.jpeg'
detect_faces_in_image(image_path)

#video_path = 'path/to/video.mp4'
#detect_faces_in_video(video_path)