import numpy as np
import cv2
import os

class WebCamera():
    """
    Initializes a new WebCamera object
    
    Args:
        haar_dir (str): path to OpenCV haarcascade xml file
    """
    def __init__(self, haar_dir='haarcascade_frontalface_default.xml'):
        self.cap = cv2.VideoCapture(0)
        self.haar_dir = haar_dir
        # Setting sizes of captured frame/window
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 450)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 330)
    
    def read_frame(self):

        ret, frame = self.cap.read(0)
        if not self.cap.isOpened():
            os.popen('open -a ScreenSaverEngine')
            raise Exception("Could not open video device")
            
        # Fliping mirror effect
        frame = cv2.flip(frame, 1)
        
        return frame

    def read_camera(self, with_face=False):

        while True:
            
            frame = self.read_frame()
            
            if with_face:
                
                frame, face = self.face_detector(frame)  

            cv2.imshow('Video Face Detection', frame)
            if cv2.waitKey(1) == 27:
                break
        self.cap.release()
        cv2.destroyAllWindows()
    
    def face_detector(self, frame):
        # detect and crop face based on the haar algorithm
        frame_copy = frame.copy()
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + self.haar_dir)
        face_rects = face_cascade.detectMultiScale(frame_copy)
        
        if len(face_rects) == 0:
            
            face = np.zeros_like(frame)
            
        for (x, y, w, h) in face_rects:
            cv2.rectangle(frame_copy, (x, y), (x + w, y + h), (255, 255, 255), 3)
            face = frame_copy[y:y+h, x:x+w]
            
        return frame_copy, face