import cv2
import tkinter as tk
from tkinter import filedialog

def detect_faces(image_path):
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    num_faces = len(faces)
    for (x, y, w, h) in faces:
        cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)

    cv2.imshow('Faces Detected', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    num_faces_label.config(text=f"Number of faces recognized: {num_faces}")

def select_image():
    image_path = filedialog.askopenfilename(initialdir='/', title='Select Image', filetypes=(('Image Files', '*.jpg *.jpeg *.png'), ('All Files', '*.*')))
    detect_faces(image_path)

window = tk.Tk()
window.title('Face Detection App')
window.geometry("400x300")
window.configure(bg='#FFD700')

frame = tk.Frame(window, bg='#FFD700')
frame.pack(expand=True)

select_button = tk.Button(frame, text='Select Image', command=select_image, bg='#FFA500', fg='white', font=('Arial', 14, 'bold'), relief='raised')
select_button.pack(pady=20)

num_faces_label = tk.Label(frame, text="Number of faces recognized: ", bg='#FFD700', font=('Arial', 12))
num_faces_label.pack()

window.mainloop()