import cv2
import data
import torch
import models
import threading
import numpy as np
import camera_module
import database_connect
from kivy.app import App
from kivy.uix.label import Label
from kivy.uix.button import Button
from kivy.uix.boxlayout import BoxLayout
from kivy.graphics.texture import Texture
from kivy.uix.gridlayout import GridLayout
from kivy.graphics import Color, Rectangle

from kivy.uix.image import Image
from kivy.clock import Clock

class KivyCamera(BoxLayout):
    def __init__(self, fps, device, model_path, haar_model='haarcascade_frontalface_default.xml',**kwargs):
        super(KivyCamera, self).__init__(**kwargs)
        self.orientation = 'horizontal'
        self.haar = haar_model
        self.cam = camera_module.WebCamera(self.haar)
        self.raw_image = Image()
        self.face = Image()
        self.device = device
        self.mse = None
        self.cos = None
        self.correlation = None
        
        # set model
        self.model_path = model_path
        ae = models.Autoencoder()
        ae.load_state_dict(torch.load(self.model_path))
        self.encoder = ae.encoder.to(self.device)
        self.encoder.eval()
        
        # images
        images_layout = BoxLayout(orientation='horizontal')  
        self.add_widget(images_layout)
        images_layout.add_widget(self.raw_image)
        images_layout.add_widget(self.face)
        
        # buttons
        button_layout = BoxLayout(orientation='horizontal', size_hint_y=0.1)
        self.add_widget(button_layout)
        
        check_id = Button(text='Check Identity')
        check_id.bind(on_press=self.check_identity_t)
        button_layout.add_widget(check_id)
        
        add_id = Button(text='Add Identity')
        add_id.bind(on_press=self.add_identity_t)
        button_layout.add_widget(add_id)
        
        # Pass/Fail
        self.pass_fail_indicator = BoxLayout(size_hint_y=0.8)
        self.pass_fail_indicator_color = (1, 1, 1, 1)  
        self.pass_fail_indicator_background = Rectangle(pos=self.pass_fail_indicator.pos, size=self.pass_fail_indicator.size)
        self.pass_fail_indicator.canvas.add(Color(*self.pass_fail_indicator_color))
        self.pass_fail_indicator.canvas.add(self.pass_fail_indicator_background)
        self.add_widget(self.pass_fail_indicator)
        
        # Results Table
        self.results_layout = GridLayout(cols=2, size_hint_y=0.2)
        self.add_widget(self.results_layout)
        
        self.results_layout.add_widget(Label(text="MSE:"))
        self.mse_label = Label(text="")
        self.results_layout.add_widget(self.mse_label)
        
        self.results_layout.add_widget(Label(text="Cosine Similarity:"))
        self.cos_label = Label(text="")
        self.results_layout.add_widget(self.cos_label)
        
        self.results_layout.add_widget(Label(text="Correlation:"))
        self.correlation_label = Label(text="")
        self.results_layout.add_widget(self.correlation_label)

        Clock.schedule_interval(self.update, 1.0 / fps)

    def update(self, dt):
        frame = self.cam.read_frame()
        if frame is not None:
            frame, face = self.cam.face_detector(frame)

            # Update camera image
            buf = cv2.flip(frame, 0).tostring()
            image_texture = Texture.create(size=(frame.shape[1], frame.shape[0]), colorfmt='bgr')
            image_texture.blit_buffer(buf, colorfmt='bgr', bufferfmt='ubyte')
            self.raw_image.texture = image_texture

            # Update face image
            buf1 = cv2.flip(face, 0).tostring()
            image_texture1 = Texture.create(size=(face.shape[1], face.shape[0]), colorfmt='bgr')
            image_texture1.blit_buffer(buf1, colorfmt='bgr', bufferfmt='ubyte')
            self.face.texture = image_texture1
            
            # Pass/Fail indicator
            if self.mse is not None:
                if self.mse < 0.00006:
                    self.pass_fail_indicator_color = (0, 1, 0, 1)  # "Pass"
                else:
                    self.pass_fail_indicator_color = (1, 0, 0, 1) # "Fail"
                self.pass_fail_indicator.canvas.clear()
                self.pass_fail_indicator.canvas.add(Color(*self.pass_fail_indicator_color))
                self.pass_fail_indicator.canvas.add(self.pass_fail_indicator_background)
            
    def add_identity_t(self, instance):
        threading.Thread(target=self.add_identity).start()
        
    def add_identity(self):
            
        frame = self.cam.read_frame()
            
        frame, face = self.cam.face_detector(frame)

        face = data.transform(face).unsqueeze(0).to(self.device)
        
        embedding = self.encoder(face)
        
        db = database_connect.DB()
        db.write(embedding)
        print(embedding)
            
    def check_identity_t(self, instance):
        threading.Thread(target=self.check_identity).start()
        
    def check_identity(self):
        
        db = database_connect.DB()
        
        embeddings = db.read()
        
        for latent_vecotor in embeddings:
            db_latent_vecotor = torch.tensor(latent_vecotor).unsqueeze(0).to(self.device)
            
        frame = self.cam.read_frame()
            
        frame, face = self.cam.face_detector(frame)

        face = data.transform(face).unsqueeze(0).to(self.device)
        
        latent_vecotor = self.encoder(face)
        
        self.mse = torch.nn.functional.mse_loss(latent_vecotor, db_latent_vecotor).item()
        print("mse:", self.mse)
        
        self.cos = torch.nn.functional.cosine_similarity(latent_vecotor, db_latent_vecotor).item()
        print("cos similarity:", self.cos)
        
        latent_vecotor, db_latent_vecotor = latent_vecotor.cpu().detach().numpy(), db_latent_vecotor.cpu().detach().numpy()
        corr = np.corrcoef(latent_vecotor, db_latent_vecotor)
        print("correlation: ", corr)
        self.correlation = corr[0, 1]
        
        # Update the labels with new values
        self.mse_label.text = f"{self.mse:.6f}"
        self.cos_label.text = f"{self.cos:.6f}"
        self.correlation_label.text = f"{self.correlation:.6f}"
        
        
class TestApp(App):
    def build(self):
        kivy_camera = KivyCamera(fps=30, device="mps", model_path="ae.pt")
        return kivy_camera

    def on_stop(self):
        # Release the capture when the app is stopped
        self.cam.release()

if __name__ == '__main__':
    TestApp().run()