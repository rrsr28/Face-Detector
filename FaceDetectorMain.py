import cv2
import tkinter as tk
from tkinter import filedialog
from PIL import ImageTk, Image

def detect_faces(image_path):

    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    for (x, y, w, h) in faces:
        cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)

    cv2.imshow('Faces Detected', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def select_image():

    image_path = filedialog.askopenfilename(initialdir='/', title='Select Image', filetypes=(('Image Files', '*.jpg *.jpeg *.png'), ('All Files', '*.*')))
    image = Image.open(image_path)
    image = image.resize((400, 400), Image.ANTIALIAS)
    
    # Convert the image to Tkinter format
    image_tk = ImageTk.PhotoImage(image)
    
    # Update the image preview in the UI
    image_preview.config(image=image_tk)
    image_preview.image = image_tk
    
    detect_faces(image_path)

window = tk.Tk()
window.title('Face Detection App')

select_button = tk.Button(window, text='Select Image', command=select_image)
select_button.pack(pady=10)

image_preview = tk.Label(window)
image_preview.pack()

window.mainloop()